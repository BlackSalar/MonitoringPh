from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import messagebox as mb
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error




window =Tk()
window.resizable(0,0)
window.title("Сбор данных системы водоотведения Барановичского КУПП \"Водоканал\"")
class Neuron:
  def __init__(self,df1,window):
    self.df1=df1
    #Меню
    mainmenu=Menu(window)
    #Создаем пункты подменю для пункта меню "Файл" 
    filemenu = Menu(mainmenu, tearoff=0)
    #Создаем еще один объект Menu
    filemenu.add_command(label="Загрузить файл", command=lambda: net.load_file())

    #Справка
    helpmenu=Menu(mainmenu, tearoff=0)
    helpmenu.add_command(label="О программе")
    #Добавляем пункты меню
    mainmenu.add_cascade(label = "Файл", menu=filemenu)
    mainmenu.add_cascade(label = "Справка", menu=helpmenu)
    #Конфигурируем окно с меню
    window.config(menu = mainmenu)
    #Прогноз Ph сточной воды. Верхний фрейм
    frame_top=Frame(window, bg='white')
    Label(frame_top, height=1, text="Прогноз ph сточной воды", font=("Times New Roman", 14),bg='white').pack()
    frame_top.pack()
    frame_top.pack(expand=0,fill=X)

    #Нижний фрейм
    frame_bottom=Frame(window)
    #Левый фрейм с графиками и RadioButtons
    frame_left=Frame(frame_bottom)

    #Правый график
    y = [i**2 for i in range(101)]
     

    #Левый график
    fig2=Figure(figsize = (5, 5), dpi = 100)
    plot2=fig2.add_subplot(111)
    plot2.set_title("Фактические данные")
    plot2.plot(y)

    plot2.set_xlabel("Datetime")
    plot2.set_ylabel("PhSensor")
    canvas2=FigureCanvasTkAgg(fig2, frame_left)
    canvas2.draw()
    canvas2.get_tk_widget().grid(row=0,column=0, padx=5,pady=5, ipadx=45)
    #Панель для работы с графиками
    toolbar = NavigationToolbar2Tk(canvas2, frame_bottom)
    toolbar.update()

    #Выбор архитектуры
    arch=ttk.Combobox(frame_left,width=30, height=1,font=("Times New Roman", 14))
    arch['values']=("Выбор архитектуры","Сверточная", "Персептрон")
    arch.current(0)
    arch.grid(row=1,column=0,padx=10,pady=10)
    #Глубина прогноза
    forecast=ttk.Combobox(frame_left,width=30, height=1,font=("Times New Roman", 14))
    forecast['values']=("Глубина прогноза", "1 час", "2 часа", "3 часа", "6 часов")
    forecast.current(0)
    forecast.grid(row=1,column=1,padx=10,pady=10)
    frame_left.pack()
    frame_left.pack(side=LEFT)

    #Правый фрейм для кнопок и вывода ошибки
    frame_right=Frame(frame_bottom)

    #Сохранить в БД
    save_db=Button(frame_right,width=30, height=1,font=("Times New Roman", 14), text="Сохранить в БД") 
    #save_db.configure(command=change)
    save_db.pack(padx=5,pady=5)
    Label(frame_right,width=30, height=1, text="Количество эпох:", font=("Times New Roman", 14),bg='white').pack(padx=5)
    def validate(new_value):                                                  # +++
     return new_value == "" or new_value.isnumeric()
    vcmd=(window.register(validate),'%P')
    epochs_numb=Entry(frame_right, width=30, validatecommand=vcmd)
    epochs_numb.pack(padx=5)
    
    frame_right_bottom=Frame(frame_right)
    #Точность моделирования НС
    Label(frame_right_bottom,width=30, height=1, text="Точность моделирования НС", font=("Times New Roman", 14),bg='white').pack(padx=5)
    Label(frame_right_bottom,width=30, height=1, text="Ошибка: 0,05", font=("Times New Roman", 14),fg='white',bg='red').pack(padx=5)
    #Обучить модель  
    learn_model=Button(frame_right,width=30, height=1,font=("Times New Roman", 14), text="Обучить модель", command=lambda: net.learn(frame_left,int(epochs_numb.get())))
    learn_model.pack(padx=5,pady=5)
    frame_right_bottom.pack()

    frame_right.pack()
    frame_right.pack()

    frame_bottom.pack()
      
  
  def load_file(self):  
   filetypes = (("Excel-файл", "*.csv"),("Любой", "*"))
   filename = fd.askopenfilename(title="Открыть файл", initialdir="/", filetypes=filetypes)   
   if filename.endswith('.csv'):
    df=pd.read_csv(filename,delimiter=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df['PHSensor'] = df['PHSensor'].str.replace(',', '.')
    df['PHSensor'] = df['PHSensor'].astype(float)
    df['ECSensor'] = df['ECSensor'].str.replace(',', '.')
    df['ECSensor'] = df['ECSensor'].astype(float)
    #df['RedoxSensor'] = df['RedoxSensor'].str.replace(',', '.')
    df['RedoxSensor'] = df['RedoxSensor'].astype(float)
    df['TPHSensor'] = df['TPHSensor'].str.replace(',', '.')
    df['TPHSensor'] = df['TPHSensor'].astype(float)    
    self.df1 = df
    FirstDate = self.df1['timestamp'][0] #получение первой даты
    five = timedelta(minutes=5) #это мы получили пять минут в переменную
    #result_df = pd.DataFrame(columns=df.columns)
    date_range = pd.date_range(start=self.df1['timestamp'].min(), end=self.df1['timestamp'].max(), freq='5T')#это создаётся как бы  массив времени с шагом в пять минут от начальной
    # Создайте DataFrame с пятью колонками и пустыми значениями
    df2 = pd.DataFrame({
     self.df1.columns[0]: date_range,
     self.df1.columns[1]: np.nan,
     self.df1.columns[2]: np.nan,
     self.df1.columns[3]: np.nan,
     self.df1.columns[4]: np.nan
    })
    self.df1 = pd.concat([self.df1, df2], ignore_index=True)#тут идёт обьеденение начального и пустого датафрейма
    self.df1 = self.df1.sort_values(by='timestamp').reset_index(drop=True)#тут сортировка идёт по времени чтобы все стало по порядку и индексы чтобы тоже были попорядку
    self.df1.bfill(inplace=True)#это заполнение пустых значений берет предыдущее доступное значение для заполнения
    self.df1 = self.df1.drop(self.df1[self.df1['timestamp'].isin(df['timestamp'])].index)#тут исключаю начальный датафрейм из нового
    self.df1 = self.df1.sort_values(by='timestamp').reset_index(drop=True)
    # Функция для фильтрации строк
    def filter_rows(row):
     # Оставляем строки, которые находятся в интервале в пять минут от FirstDate
     return (row['timestamp'] - FirstDate) % five == pd.Timedelta(0)
    df_filtered = df[df.apply(filter_rows, axis=1)]# тут мы из первого датафрейма берём те данные которые как бы с шагом в пять минут попадали и до обработки таких точек было может тысяча
    self.df1 = pd.concat([self.df1, df_filtered], ignore_index=True)#тут соединяю и фильтрую и все
    self.df1 = self.df1.sort_values(by='timestamp').reset_index(drop=True)
    
    self.df1.index = pd.to_datetime(self.df1['timestamp'], format='%Y-%m-%d %H:%M:%S') #format='%Y-%m-%d %H:%M:%S'
    del self.df1['timestamp']        
    mb.showinfo("Загрузка","Файл загружен и преобразован")    
   else:
    mb.showwarning("Ошибка","Нужен файл формата .csv")

   
  def learn(self,frame_left,epochs_numb):
    try:
      # Extract the data you want to use for LSTM (e.g., 'PHSensor' column)
      data = self.df1['PHSensor'].values
      data = data.reshape(-1, 1)  # Reshape the data to have a single feature
      print(data)
      # Normalize the dataset
      scaler = MinMaxScaler(feature_range=(0, 1))
      data = scaler.fit_transform(data)
      # Split the dataset into train and test sets
      train_size = int(len(data) * 0.8)
      test_size = len(data) - train_size
      train, test = data[0:train_size, :], data[train_size:len(data), :]
      # Function to create the dataset for LSTM
      def create_dataset(dataset, look_back=1):
       dataX, dataY = [], []
       for i in range(len(dataset) - look_back - 1):
          a = dataset[i:(i + look_back), 0]
          dataX.append(a)
          dataY.append(dataset[i + look_back, 0])
       return np.array(dataX), np.array(dataY)
      # Reshape input to be [samples, time steps, features]
      look_back = 60
      trainX, trainY = create_dataset(train, look_back)
      testX, testY = create_dataset(test, look_back)

      trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
      testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
      # Create and fit the LSTM network
      model = Sequential()
      model.add(LSTM(4, input_shape=(1, look_back)))
      model.add(Dense(1))
      model.compile(loss='mean_squared_error', optimizer='adam')
      model.fit(trainX, trainY, epochs=epochs_numb, batch_size=256, verbose=1)
      # Make predictions
      trainPredict = model.predict(trainX)
      testPredict = model.predict(testX)
      # Invert predictions
      trainPredict = scaler.inverse_transform(trainPredict)
      trainY = scaler.inverse_transform([trainY])
      testPredict = scaler.inverse_transform(testPredict)
      testY = scaler.inverse_transform([testY])
      # Calculate root mean squared error
      trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
      print('Train Score: %.2f RMSE' % (trainScore))
      testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
      print('Test Score: %.2f RMSE' % (testScore))
      # Shift train predictions for plotting
      trainPredictPlot = np.empty_like(data)
      trainPredictPlot[:, :] = np.nan
      trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
      # Shift test predictions for plotting
      testPredictPlot = np.empty_like(data)
      testPredictPlot[:, :] = np.nan
      testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(data) - 1, :] = testPredict
      mb.showinfo("Обучение","Модель обучена")
      
      fig = Figure(figsize = (5, 5), dpi = 100)
       
      ax1 = fig.add_subplot(111)
      ax1.set_title("Прогноз Ph")
      ax1.set_xlabel("Datetime")
      ax1.set_ylabel("PhSensor")
      ax1.plot(self.df1.index[160:220],scaler.inverse_transform(data[160:220]))
      ax1.plot(self.df1.index[160:220],trainPredict[100:160], color='red')
      canvas = FigureCanvasTkAgg(fig, frame_left)  
      canvas.draw()  
      canvas.get_tk_widget().grid(row=0,column=1, padx=5,pady=5, ipadx=45)
      
      fig_loss = Figure(figsize = (5, 5), dpi = 100)
       
      ax2 = fig_loss.add_subplot(111)
      ax2.set_title("График ошибки")
      ax2.set_xlabel("Epochs")
      ax2.set_ylabel("loss")
      
      
      canvas2 = FigureCanvasTkAgg(fig_loss, frame_left)  
      canvas2.draw()  
      canvas2.get_tk_widget().grid(row=0,column=2, padx=5,pady=5, ipadx=45)
      
          
    except:
      mb.showwarning("Ошибка", "Модель не обучена.")
df1=pd.DataFrame()




net=Neuron(df1,window)

window.mainloop()