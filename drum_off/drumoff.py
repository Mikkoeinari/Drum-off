import importlib
importlib.invalidate_caches()

from drum_off import UI

if __name__ == "__main__":
    #from drum_off import UI
    #UI.mainthread()

    UI.DrumOffApp().run()
