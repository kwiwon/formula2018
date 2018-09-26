# Intro
* 大部分都是以當初ti6的code為基礎改的
* base model 還是VGG16 (訓練得完)
* Train出來的model可以跟 Lawerance 的 drive.py 一起使用

# More than ti6...
* ti6版本的資料格式是直接使用sdk輸出的，這裡改成使用ti6給的[dataset](https://www.tbox.trend.com.tw/app#folder/STbrN/AI_Car_Q_team_TOI/driving-records-20180906.zip?a=Rtpzjn231pA)，基本上把整包放置到behavioral_cloning/drive/下面即可
* 如果要追加資料也可以，但目前試的結果來說，感覺上以ti6的資料集去做重點訓練會比我們自己追加資料來的有效率。（個人感覺，不確定）
* 關於增加資料比例的部分，可以參考目前train.py。目前採取兩個方式去增加資料比例，一個是如果看到畫面上出現一定比例的黃色，就判定這筆資料可能是在三線道上面跑的。另外則是steering的數值，如果取絕對值後超過一定範圍，則認定這筆資料是在彎道上。這部分可以考慮做更細膩一點的處理，例如每一段的範圍的資料比例都不一樣。
* 另外 augmentation.py 則是一些可以增加資料多樣性的function，可以斟酌使用（但他是rgb，記得要轉換回bgr）。如果有使用到這邊的function，整體收斂時間會變慢很多。
* CNN的全連接層有更動過一點數值，另外關於epoch的部分，預設值改為10000，收斂完成後會自動結束，不會真的跑完10000回合。