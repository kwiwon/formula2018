#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <math.h>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"
#include <fstream>
#include <typeinfo>
using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

using CppAD::AD;

bool debug_mode = false;

size_t N = 15;
double dt = 0.08;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
double Lf = 2.67;
double latency = 0.1;
double MAX_ACCELERATION = 1;
double MAX_STEERING_ANGLE = 25;

double ref_cte = 0;
double ref_epsi = 0;
double ref_v = 80;

double weight_cte = 100;
double weight_epsi = 100;
double weight_speed = 1;
double weight_actuator_steering = 100000;
double weight_actuator_throttle = 50;
double weight_steering_change = 10000;
double weight_throttle_change = 10;

size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); i++) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
    assert(xvals.size() == yvals.size());
    // assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);

    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }

    for (int j = 0; j < xvals.size(); j++) {
        for (int i = 0; i < order; i++) {
            A(j, i + 1) = A(j, i) * xvals(j);
        }
    }

    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    // Setup cost (fg[0]) to be optimized
    fg[0] = 0;

    // Cost for reference state
    for (int i = 0; i < N; i++) {
      fg[0] += weight_cte * CppAD::pow(vars[cte_start + i] - ref_cte, 2);
      fg[0] += weight_epsi * CppAD::pow(vars[epsi_start + i] - ref_epsi, 2);
      fg[0] += weight_speed * CppAD::pow(vars[v_start + i] - ref_v, 2);
    }

    // Minimize actuator use
    for (int i = 0; i < N - 1; i++) {
      fg[0] += weight_actuator_steering * CppAD::pow(vars[delta_start + i], 2);
      fg[0] += weight_actuator_throttle * CppAD::pow(vars[a_start + i], 2);
    }

    // Minimize actuator change
    for (int i = 0; i < N - 2; i++) {
      fg[0] += weight_steering_change * CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i] , 2);
      fg[0] += weight_throttle_change * CppAD::pow(vars[a_start + i + 1] - vars[a_start + i] , 2);
    }

    //
    // Setup Constraints
    //

    // Initial state contraints are the same as state0
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (size_t t = 1; t < N; t++) {
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> a0 = vars[a_start + t - 1];

      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * CppAD::pow(x0, 2)
          + coeffs[3] * CppAD::pow(x0, 3);
      AD<double> psides0 = CppAD::atan(coeffs[1]
          + 2 * x0 * coeffs[2] + 3 * coeffs[3] * CppAD::pow(x0, 2));

      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t] =
          cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t] =
          epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);

    }

  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

void MPC::UpdateSettings(string config_path) {
    // Read model settings from file
    ifstream conf(config_path);
    string settings_str((istreambuf_iterator<char>(conf)), istreambuf_iterator<char>());
    auto model_settings = json::parse(settings_str);

    debug_mode = model_settings["debug"];
    N = model_settings["num_predict_steps"];
    dt = model_settings["step_duration"];
    Lf = model_settings["Lf"];
    latency = model_settings["latency"];
    MAX_ACCELERATION = model_settings["max_acceleration"];
    MAX_STEERING_ANGLE = model_settings["max_steering_angle"];
    ref_cte = model_settings["ref_cte"];
    ref_epsi = model_settings["ref_epsi"];
    ref_v = model_settings["ref_speed"];
    weight_cte = model_settings["weight"]["cte"];
    weight_epsi = model_settings["weight"]["epsi"];
    weight_speed = model_settings["weight"]["speed"];
    weight_actuator_steering = model_settings["weight"]["actuator_steering"];
    weight_actuator_throttle = model_settings["weight"]["actuator_throttle"];
    weight_steering_change = model_settings["weight"]["steering_change"];
    weight_throttle_change = model_settings["weight"]["throttle_change"];

    x_start = 0;
    y_start = x_start + N;
    psi_start = y_start + N;
    v_start = psi_start + N;
    cte_start = v_start + N;
    epsi_start = cte_start + N;
    delta_start = epsi_start + N;
    a_start = delta_start + N - 1;
}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];

  // Set the number of model variables (includes both states and inputs).
  // (x, y, psi, v, cte, e_psi), (delta, a)
  size_t n_vars = N * 6 + (N - 1) * 2;
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // Set all non actuator upper and lower limits
  // to the max negative and positive values.
  for (i = 0; i< delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] =  1.0e19;
  }

  for (i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -deg2rad(MAX_STEERING_ANGLE);
    vars_upperbound[i] =  deg2rad(MAX_STEERING_ANGLE);
  }

  // Set acceleration limit
  for (i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -MAX_ACCELERATION;
    vars_upperbound[i] =  MAX_ACCELERATION;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;
  
  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          100\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  if (debug_mode) {
      std::cout << "cost:" << cost << std::endl;
  }

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  vector<double> result;



  for (size_t i=0; i < N-1; i++) {
      result.push_back(solution.x[delta_start + i]);
      result.push_back(solution.x[a_start + i]);
  }

  for (size_t i = 0; i < N-1; i++) {
    result.push_back(solution.x[x_start + i]);
    result.push_back(solution.x[y_start + i]);
  }

  return result;
}

string MPC::Predict(string s) {
    // Response json message
    json msgJson;

    try {
        auto j = json::parse(s);
        vector<double> ptsx = j["ptsx"];
        vector<double> ptsy = j["ptsy"];
        double v = j["speed"];

        double x = 0;
        double y = 0;
        double psi = 0;

        // The input reference line should be in car coordinate space
        // Convert to Eigen::VectorXd
        double *ptrx = &ptsx[0];
        Eigen::Map<Eigen::VectorXd> xvals(ptrx, ptsx.size());

        double *ptry = &ptsy[0];
        Eigen::Map<Eigen::VectorXd> yvals(ptry, ptsy.size());

        auto coeffs = polyfit(xvals, yvals, 3);
        double cte = polyeval(coeffs, 0);
        // epsi was:
        //   psi0 - atan(coeffs[1] + 2 * x0 * coeffs[1] + 3 * coeffs[2] * pow(x0, 2))
        // where x0 = 0, psi0 = 0
        double epsi = -atan(coeffs[1]);

        Eigen::VectorXd x0(6);
        // TODO consider latency, use predict values
        x0 << x, y, psi, v, cte, epsi;

        auto solution = Solve(x0, coeffs);

        // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
        // Otherwise the values will be in between [-deg2rad(25), deg2rad(25)] instead of [-1, 1].

        // double steer_value = solution[0] / (deg2rad(25)*Lf);
        double steer_value = solution[0] / deg2rad(25);
        double throttle_value = solution[1];
        msgJson["steering_angle"] = -steer_value;
        msgJson["throttle"] = throttle_value;

        vector<double> next_steering_vals;
        vector<double> next_throttle_vals;

        for (size_t i = 2; i < N * 2; i += 2) {
            next_steering_vals.push_back(solution[i]);
            next_throttle_vals.push_back(solution[i + 1]);
        }

        msgJson["next_steering"] = next_steering_vals;
        msgJson["next_throttle"] = next_throttle_vals;

        //Display the waypoints/reference line
        vector<double> next_x_vals;
        vector<double> next_y_vals;

        //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
        // the points in the simulator are connected by a Yellow line

        int num_points = 25;
        double ploy_inc = 2.5;
        for (size_t i = 1; i < num_points; i++) {
            next_x_vals.push_back(i * ploy_inc);
            next_y_vals.push_back(polyeval(coeffs, i * ploy_inc));
        }

        msgJson["next_x"] = next_x_vals;
        msgJson["next_y"] = next_y_vals;


        //Display the MPC predicted trajectory
        vector<double> mpc_x_vals;
        vector<double> mpc_y_vals;

        //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
        // the points in the simulator are connected by a Green line

        double predict_x = v * latency;
        double predict_y = 0;
        for (size_t i = N * 2; i < solution.size(); i += 2) {
            mpc_x_vals.push_back(solution[i] + predict_x);
            mpc_y_vals.push_back(solution[i + 1] + predict_y);
        }

        msgJson["mpc_x"] = mpc_x_vals;
        msgJson["mpc_y"] = mpc_y_vals;
    } catch (...) {
        cout << "Exception in Predict function" << endl;
    }


    auto msg = msgJson.dump();
    return msg;
};

extern "C" {
    MPC mpc;

    void ChangeSettings(char* path){
        string config_file(path);
        mpc.UpdateSettings(config_file);
    }

    char *Predict(char* s) {
        string input_str(s);
        string res;
        res = mpc.Predict(input_str);
        char* res_char_p = strdup(res.c_str());

        return res_char_p;
    }
}
