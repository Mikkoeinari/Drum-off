import kivy

kivy.require('1.10.1')
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ListProperty
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooser
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import StringProperty
from kivy.clock import Clock, mainthread
import drumoff
import utils
from os import mkdir, listdir
import threading
from os.path import join, isdir
import time
#from drumsynth import playFile
import drumsynth
import GRU as mgu
# import GRU


class CCheckBox(CheckBox):
    pass


Factory.register('CCheckBox', cls=CCheckBox)
#model=mgu.initModel()

drumNames = {'kick': 0,  # only one allowed
             'snare': 1,  # only one allowed
             'closed hi-hat': 2,  # only one allowed
             'open hi-hat': 3,  # only one allowed
             'hi-hat pedal': 4,  # only one allowed
             'rack tom': 5,  # leave space for more
             'floor tom': 9,  # leave space for more
             'ride cymbal': 12,  # only one allowed
             'crash cymbal': 13,  # leave space for more
             'misc': 20
             }


class StartScreen(Screen):
    def getStatus(self):
        app = App.get_running_app()
        string = 'Current Kit:{} (kit size:{} drums)\n'.format(app.KitName, app.NrOfDrums) + 'Kit Ready:{}\n'.format(
            app.KitInit) + 'Model ready:{}'.format(app.Model)
        return string

    def SetText(self):
        app = App.get_running_app()
        if len(app.NrOfDrums):
            nr = sum(app.NrOfDrums)
        else:
            nr = 0
        string = 'Current Kit:{} (kit size:{} drums)\n'.format(app.KitName, nr) + 'Kit Ready:{}\n'.format(
            app.KitInit) + 'Model ready:{}'.format(app.Model)
        self.ids.status.text = string

    pass


class SoundCheckScreen(Screen):
    def saveKitTemplate(self, instance):
        app = App.get_running_app()
        app.KitName = self.ids.drumkit_name.text
        app.NrOfDrums = [int(self.ids.kdn.text), int(self.ids.snn.text), int(self.ids.hhn.text),
                         int(self.ids.ttn.text), int(self.ids.rdn.text), int(self.ids.crn.text), int(self.ids.ton.text)]
        # self.manager.get_screen('MainMenu').SetText()
        # print(app.KitName)


class SoundCheckPerformScreen(Screen):
    performMessage = StringProperty()
    btnMessage = StringProperty()
    drmMessage = StringProperty()
    nrMessage = StringProperty()
    finishMessage = StringProperty()
    finishStatus = StringProperty()

    def __init__(self, **kwargs):
        super(SoundCheckPerformScreen, self).__init__(**kwargs)
        self.performMessage = 'Next Drum:'
        self.btnMessage = 'GO!'
        self.drmMessage = 'Play 32 hits on each drum. \nPress run when ready!\n (You can re run souncheck as many times as you like)'
        self.nrMessage = '1'
        self.finishMessage = 'Load Soundchecked Kit and Exit'
        self.finishStatus = 'Soundcheck not complete'

    @mainthread
    def getTxt(self):
        app = App.get_running_app()
        return str(app.NrOfDrums)

    @mainthread
    def setMsg(self, msg):
        self.performMessage = msg

    @mainthread
    def setDrumNro(self, nr):
        app = App.get_running_app()
        if nr <= sum(app.NrOfDrums):
            self.nrMessage = nr
        else:
            self.ids.checkBtn.txt = 'Finished'
            self.nrMessage = ''

    # @mainthread
    def getDrumNro(self):
        return int(self.ids.drumNro.text)

    def runSoundCheck(self, nr):
        if nr == '' or nr == '0':
            utils._ImRunning = False
            return
        self.btnMessage = 'NEXT DRUM!'
        app = App.get_running_app()
        print(app.KitName, sum(app.NrOfDrums))
        fullPath = './Kits/{}'.format(app.KitName)
        try:
            mkdir(fullPath)
        except Exception as e:
            pass
        utils._ImRunning = False
        #Should we have a count in?
        time.sleep(1)
        self.performMessage = 'Start Playing!'

        def callback():
            try:
                drumoff.soundcheckDrum(fullPath, self.getDrumNro()-1)
            except Exception as e:
                print('sounckech ui:',e)

        t = threading.Thread(target=callback)
        t.start()
        self.nrMessage = str(self.getDrumNro() + 1)
        self.performMessage = 'Next Drum:'
        if (int(self.nrMessage) >= sum(app.NrOfDrums) + 1):
            self.btnMessage = 'Finish'
            self.finishStatus = 'Soundcheck Complete\nPress "Finish Souncheck" to load kit and exit'
            self.nrMessage = str(0)

    def finishSoundCheck(self):
        app = App.get_running_app()
        fullPath = './Kits/{}'.format(app.KitName)
        try:
            drumoff.initKitBG(fullPath, sum(app.NrOfDrums))
            app.KitInit = 'Initialized'
            app.root.current = 'MainMenu'
        except Exception as e:
            print('finish soundcheck: ',e)

    def First_Thread(self, nr):
        self.setMsg('soundcheck drum nr. {}'.format('0'))
        threading.Thread(target=self.runSoundCheck(nr)).start()


class PlayScreen(Screen):
    performMessage = StringProperty()
    pBtnMessage = StringProperty()
    cBtnMessage = StringProperty()
    turnMessage = StringProperty()
    lastMessage = StringProperty()
    lastGenPart = StringProperty()
    playBackMessage = StringProperty()
    def __init__(self, **kwargs):
        super(PlayScreen, self).__init__(**kwargs)
        self.performMessage = 'Press play to start'
        self.pBtnMessage = 'Play'
        self.turnMessage = 'Player'
        self.lastMessage = ''
        self.lastGenPart = ''
        self.playBackMessage = ''

    def playBackLast(self):
        if(self.lastMessage==''):
            return None
        fullPath = self.lastMessage
        if self.playBackMessage=='Play Back Computer Performance':
            fullPath=self.lastGenPart
        if self.playBackMessage=='Stop Playback':
            drumsynth._imRunning=False
            if self.turnMessage=='computer':
                self.playBackMessage = 'Play Back Computer Performance'
            else:
                self.playBackMessage = 'Play Back Player Performance'
            return None
        def callback():
            try:
                self.playBackMessage = 'Stop Playback'
                drumsynth.playFile(fullPath)
            except Exception as e:
                print('playback ui: ',e)
        t = threading.Thread(target=callback)
        t.start()

    def doComputerTurn(self):
        if (self.lastMessage == ''):
            return None
        fullPath = self.lastMessage
        def callback():
            try:##INIT MODEL!!
                self.lastGenPart = mgu.generatePart(mgu.train(fullPath, sampleMul=1.33, forceGen=False))
                self.playBackMessage = 'Play Back Computer Performance'
                self.turnMessage=('computer')
            except Exception as e:
                print('virhe! computer turn: ',e)
        t = threading.Thread(target=callback)
        t.start()
        print ('c.puter')

    def doPlayerTurn(self):
        app = App.get_running_app()
        print(app.KitName, sum(app.NrOfDrums))
        fullPath = './Kits/{}'.format(app.KitName)
        utils._ImRunning = False
        try:
            mkdir('{}/takes/'.format(fullPath))
        except Exception as e:
            pass

        if self.pBtnMessage == 'Stop':
            self.btnMessage = 'Play'
            return
        else:
            self.pBtnMessage = 'Stop'
            self.turnMessage='Computer'
        self.performMessage = 'Start Playing!'
        time.sleep(1)
        def callback():
            try:
                filename=drumoff.playLive(fullPath)
                if filename==False:
                    self.turnMessage = ('player')
                    self.pBtnMessage = 'Play'
                    return
                self.lastMessage=filename
                self.playBackMessage = 'Play Back Last Performance'
                self.turnMessage=('player')
                self.pBtnMessage = 'Play'
            except Exception as e:
                print('player playback ui: ',e)
        t = threading.Thread(target=callback)
        t.start()



class LoadScreen(Screen):

    def is_dir(self, directory, filename):
        return isdir(join(directory, filename))

    def getKits(self):
        return 'eka\ntoka\nkolmas\njne...'

    def loadKit(self, filename):
        try:

            drumoff.loadKit(filename[0])
            app = App.get_running_app()
            app.KitName = (filename[0].split('/')[-1])
            app.KitInit = 'Initialized'
            app.NrOfDrums = [0] * len(drumoff.drums)
            for i in drumoff.drums:
                drum = i.get_name()[0]
                # make hihats one drum
                app.NrOfDrums[drum] += 1
            app.root.current = 'MainMenu'
        except Exception as e:
            print('load kit ui: ',e)


class DrumOffApp(App):
    Builder.load_file("UI.kv")
    NrOfDrums = []
    KitName = 'none'
    Model = 'model not loaded'
    KitInit = 'Kit not initialized'

    def build(self):
        sm = ScreenManager()
        # All ui views
        sm.add_widget(StartScreen(name='MainMenu'))
        sm.add_widget(SoundCheckScreen(name='SoundCheck'))
        sm.add_widget(SoundCheckPerformScreen(name='SoundCheckPerform'))
        sm.add_widget(PlayScreen(name='PlayScreen'))
        sm.add_widget(LoadScreen(name='LoadKit'))
        ###LOAD MODEL HERE???###
        return sm


if __name__ == "__main__":
    Builder.load_file("UI.kv")
    DrumOffApp().run()
