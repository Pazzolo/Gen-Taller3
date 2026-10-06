import json, time, threading
import numpy as np
hijo = threading.Thread(target=time.sleep, args=(600,))
hijo.start()
time.sleep(600)
json.dump({'nunca': 1}, open('resultados.json', 'w'))
