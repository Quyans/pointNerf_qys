import time

localtime = time.localtime(time.time())
secs = time.mktime(localtime)
print("secs",secs)

while True:
    localtime1 = time.localtime(time.time())
    secs1 = time.mktime(localtime1)
    if (secs1-secs)>5:
        break;

print("出来了")
