#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod


class Car(object):
    @abstractmethod
    def on_dashboard(self, data):
        """
        :return: A tuple of steering_angle, throttle
        """
        raise NotImplementedError
