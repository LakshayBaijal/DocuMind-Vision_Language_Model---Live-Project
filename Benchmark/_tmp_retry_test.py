import time
import httpx
url = 'https://upload.wikimedia.org/wikipedia/commons/e/e8/1920_Palestine_-_EEF_-_laissez_passer.jpg'
with httpx.Client(timeout=60.0, headers={'User-Agent':'Mozilla/5.0'}) as c:
    for i in range(1,9):
        r=c.get(url)
        print(i, r.status_code, r.headers.get('retry-after'))
        if r.status_code==200:
            print('ok bytes',len(r.content))
            break
        time.sleep(min(20, i*2))
