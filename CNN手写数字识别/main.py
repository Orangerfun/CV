from tkinter import *
from tkinter.filedialog import *
from PIL import ImageTk
import ccn
import cv2
import os

def show_rec(filename):
	img=ImageTk.PhotoImage(file=filename)
	Label(root,image=img).grid(row=4,pady=15)
	result=ccn.main(filename)
	Label(root,text=str(result),font=("微软雅黑",20)).grid(row=5,column=1)
	mainloop()

def select_img():
	filename=askopenfilename(defaultextension=".jpg",title="选择图片",filetypes=[("JPG",".jpg")])
	show_rec(filename)

def openvideo():
	cap=cv2.VideoCapture(0)
	while True:
		ret,frame=cap.read()
		if ret==True:
			cv2.imshow("video",frame)
			if cv2.waitKey(1)==ord("q"):
				break
			if cv2.waitKey(25)==ord("s"):
				img=cv2.resize(frame,(28,28))
				cv2.imwrite("img.jpg",img)
				break
	cap.release()
	cv2.destroyAllWindows()
	show_rec("./img.jpg")

def exit_pro():
	os._exit(0)

if __name__=="__main__":
	root=Tk()
	root.geometry('650x450+500+200')
	root.title("手写数字识别")
	Button(root,text="选择图片",command=select_img,font=("微软雅黑",20),bg="#1E90FF").grid(row=0,column=0,padx=30)
	Button(root,text="打开相机",command=openvideo,font=("微软雅黑",20),bg="#6495ED").grid(row=0,column=1,padx=30)
	Button(root,text="退出程序",command=exit_pro,font=("微软雅黑",20),bg="#87CEFA").grid(row=0,column=2,padx=30)
	Label(root,text="已捕获图片:",font=("微软雅黑",20)).grid(row=2,pady=15)
	Label(root,text="预测结果:",font=("微软雅黑",20)).grid(row=5,pady=15)
	Label(root,text="注意：选择【打开相机】看见图像流后按键盘S键开始捕获图像并识别",font=("宋体",10),fg="red").grid(row=1,pady=15,columnspan=3)
	mainloop()