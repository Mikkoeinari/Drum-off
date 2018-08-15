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

Builder.load_file("UI.kv")
NrOfDrums=0
KitName='none'
Model='model not loaded'
KitInit='Kit not initialized'
def getGameStatus(KitName=KitName, KitInit=KitInit, Model=Model, NrOfDrums=NrOfDrums):
    string='Current Kit:{}\n'.format(KitName)+'Kit Ready:{}\n'.format(KitInit)+'Model ready:{}'.format(Model)
    print(NrOfDrums)
    return string

def setKitName(name):
    KitName=str(name)
    print('name changed', KitName)
class CCheckBox(CheckBox):
    pass

Factory.register('CCheckBox', cls=CCheckBox)
class StartScreen(Screen):
    def getStatus(self):
        return getGameStatus('None', 'None', 'None')
    def SetText(self,msg):
        KitName=msg
        self.ids.status.text=msg
    pass

class SoundCheckScreen(Screen):
    def saveKitTemplate(self,instance):
        setKitName(self.ids.txt.text)
        NrOfDrums=(int(self.ids.kdn.text)+int(self.ids.snn.text)+int(self.ids.hhn.text)+int(self.ids.ttn.text)+int(self.ids.rdn.text)+int(self.ids.crn.text)+int(self.ids.ton.text))
        self.manager.get_screen('MainMenu').SetText(getGameStatus(KitName=self.ids.txt.text, NrOfDrums=NrOfDrums))


class SoundCheckPerformScreen(Screen):
    text='hehehe'
    def getTxt(self):
        return str(NrOfDrums)

class PlayScreen(Screen):
    pass

class LoadScreen(Screen):
    def getKits(self):
        return 'eka\ntoka\nkolmas\njne...'
    def loadKit(self,filename):
        print(filename)

sm = ScreenManager()
#All ui views
sm.add_widget(StartScreen(name='MainMenu'))
sm.add_widget(SoundCheckScreen(name='SoundCheck'))
sm.add_widget(SoundCheckPerformScreen(name='SoundCheckPerform'))
sm.add_widget(PlayScreen(name='PlayScreen'))
sm.add_widget(LoadScreen(name='LoadKit'))

class DrumOffApp(App):

    def build(self):
        return sm


if __name__ == "__main__":
    DrumOffApp().run()