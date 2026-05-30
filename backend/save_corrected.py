import requests, base64
r = requests.post('http://127.0.0.1:8000/correct?method=jaw', files={'file': open('test_photo.jpg','rb')})
print('status', r.status_code)
data = r.json()
for name in ('before','mesh','after','after_graph'):
    val = data.get(name,'')
    if val:
        if val.startswith('data:image/png;base64,'):
            b = base64.b64decode(val.split(',',1)[1])
        else:
            b = base64.b64decode(val)
        open(f'{name}.png','wb').write(b)
        print('wrote', name+'.png', len(b))
    else:
        print('no', name)
