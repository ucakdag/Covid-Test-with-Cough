
###UMUT CAN AKDAG



#import pyaudio
import wave
import librosa 
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import KFold
import noisereduce as nr
from sklearn import metrics
import pickle
from scipy.io import wavfile
from scipy import signal
import pyaudio
import wave
import os.path
import string    
import random

path1 = 'C:/Users/Lenovo/Desktop/IOT/covid-19/'
path2 = 'C:/Users/Lenovo/Desktop/IOT/healthy/'
path3 = 'C:/Users/Lenovo/Desktop/IOT/data_file/'

coughh = []

p1,p3,p4c = 0,0,0

for filename in os.listdir(path1):
        audio_data, sample_rate = librosa.load(path1 + filename,mono = True,sr = None)
        #sample_rate, audio_data = wavfile.read(path1 + filename)
        reduced = nr.reduce_noise(audio_data, sample_rate)
        stft = np.abs(librosa.stft(reduced,n_fft = 2058))
        #stft = np.abs(signal.stft(reduced, sample_rate, nfft = 1024))
        dfStft = pd.DataFrame(stft)
        dfStftCut = dfStft.iloc[50:200]
        arr = dfStftCut.to_numpy()
        stftMean = np.mean(arr,axis=1)
        coughh.append(stftMean)
        p1=p1+1

coughhDF = pd.DataFrame(coughh)
coughhDF.shape



mfcss_notcrying_p2= []

p2 = 0

for filename in os.listdir(path2):
        audio_data, sample_rate = librosa.load(path2 + filename,mono = True,sr = None)
        #sample_rate, audio_data = wavfile.read(path2 + filename)
        reduced = nr.reduce_noise(audio_data, sample_rate)
        stft = np.abs(librosa.stft(reduced,n_fft = 2058))
        #stft = np.abs(signal.stft(reduced, sample_rate, nfft = 1024))
        dfStft = pd.DataFrame(stft)
        dfStftCut = dfStft.iloc[50:200]
        arr = dfStftCut.to_numpy()
        stftMean = np.mean(arr,axis=1)
        mfcss_notcrying_p2.append(stftMean)
        p2=p2+1

mfcss_notcrying_p2DF = pd.DataFrame(mfcss_notcrying_p2)
mfcss_notcrying_p2DF.shape



coughhDF["status"] = 1
mfcss_notcrying_p2DF["status"] = 0

df = pd.concat([coughhDF, mfcss_notcrying_p2DF])
df.shape
dfSonuc = df.iloc[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size=0.25,random_state=1907)
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
while(True):
    S = 10
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = S))
     
    def play_sound(path_to_file):

       print ("Now Playing...")
       chunk = 1024
       f = wave.open(path_to_file)
       p = pyaudio.PyAudio()

       stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                       channels=f.getnchannels(),
                       rate=f.getframerate(),
                       output=True)

       data = f.readframes(chunk)

       for i in range(len(data)):
           stream.write(data)
           data = f.readframes(chunk)
    
       stream.stop_stream()
       stream.close()
    
       p.terminate()
    
    
    def record_my_audio():
       time_duration = int(input('Time Length'))
       FORMAT = pyaudio.paInt16
       CHANNELS = 2
       RATE = 44100
       CHUNK = 1024
       RECORD_SECONDS = time_duration
       print ("Recording...")
       audio = pyaudio.PyAudio()
       stream = audio.open(format=FORMAT, channels=CHANNELS,
                           rate=RATE, input=True,
                           frames_per_buffer=CHUNK)
       frames = []

       for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
           data = stream.read(CHUNK)
           frames.append(data)

       stream.stop_stream()
       stream.close()
       audio.terminate()
       print ("Recorded\n")
       return_data = [frames, stream, audio]
       return return_data

    def save_my_recording(destination_filename, stream, frames, audio):
       channels = stream._channels
       rate = stream._rate
       format = stream._format

       wave_File = wave.open(destination_filename, 'wb')
       wave_File.setnchannels(channels)
       wave_File.setsampwidth(audio.get_sample_size(format))
       wave_File.setframerate(rate)
       wave_File.writeframes(b''.join(frames))
       wave_File.close()
    def get_user_input():
       print ("What do you want to do :\n 1) Record audio\n")
       entered_value = int(input("Enter 1 to record audio\nEnter 2 to analiz past recording"))
       if entered_value in [1, 2]:
            
            return entered_value
       else:
            print ("Try Again. \n")
            return get_user_input()
    choice = get_user_input()
    def get_user_mail():
        
        entered_mail = str(input("Enter your mail adress"))
        return entered_mail
    data = record_my_audio()
    maill= get_user_mail()   
    if choice == 1:

        file_name = str(ran) +".wav"
        save_my_recording( path3 + file_name, data[1], data[0], data[2])
    
    elif (choice == 2):
        file_name = input("name recording :") +".wav"
    
    
    ################################################################################################
    
    audio_data3, sample_rate3 = librosa.load(path3 + file_name,mono = True,sr = None)
    deneme3 = nr.reduce_noise(audio_data3,sample_rate3)
    stft3 = np.abs(librosa.stft(deneme3,n_fft = 2058))
    dfStft3 = pd.DataFrame(stft3)
    dfStftCut3 = dfStft3.iloc[50:200]
    arr3 = dfStftCut3.to_numpy()
    stftMean3 = np.mean(arr3,axis=1)
    
    
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict([stftMean3])

    if(y_pred == 1):
        mail = smtplib.SMTP("smtp.gmail.com",587)          
        mail.ehlo()
        mail.starttls()

        mail.login("covidsonucu@gmail.com", "oksuruksesi") 
        mesaj = MIMEMultipart()
        mesaj["From"] = "covidsonucu@gmail.com"        # Gönderen kişi
        mesaj["To"] = maill        # Alıcı

        mesaj["Subject"] = "Öksürük Sesinizin Sonucu "  # Konu
    
        body = """POZITIF, PCR TESTI ICIN EN YAKIN HASTANEYE BASVURUN
        """
        
        body_text = MIMEText(body, "plain")  
        mesaj.attach(body_text)
        mail.sendmail( mesaj["From"], mesaj["To"], mesaj.as_string())
        print("SONUCUNUZ MAIL ADRESINIZE ILETILDI")
        print("Covid-19")
    else: 
        mail = smtplib.SMTP("smtp.gmail.com",587)          
        mail.ehlo()
        mail.starttls()

        mail.login("YOURMAIL@gmail.com", "YOUR MAIL PASSWORD") 
        mesaj = MIMEMultipart()
        mesaj["From"] = "covidsonucu@gmail.com"      
        mesaj["To"] = maill          

        mesaj["Subject"] = "Öksürük Sesinizin Sonucu "
    
        body = """

        NEGATIF, DAHA NET BILGI SAHIBI OLMAK ADINA LUTFEN PCR TESTI ICIN EN YAKIN HASTANEYE BASVURUN

        """
        
        body_text = MIMEText(body, "plain")  
        mesaj.attach(body_text)
        mail.sendmail( mesaj["From"], mesaj["To"], mesaj.as_string())
        print("SONUCUNUZ MAIL ADRESINIZE ILETILDI")        
        print("Healthy")
    
    y_pred2 = clf.predict(X_test)
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))
    print("Precision:",metrics.precision_score(y_test, y_pred2))
    print("Recall:",metrics.recall_score(y_test, y_pred2))
    print(metrics.classification_report(y_test, y_pred2))