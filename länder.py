import os.path

import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urlparse
import re

# Initiale Listen
zu_besuchend = ["http://gov" + el for el in ".ac,.ad,.ae,.af,.ag,.ai,.al,.am,.ao,.aq,.ar,.as,.at,.au,.aw,.ax,.az,.ba,.bb,.bd,.be,.bf,.bg,.бг,.bh,.bi,.bj,.bm,.bn,.bo,.bq,.br,.bs,.bt,.bw,.by,.bz,.st,.ca,.cc,.cd,.cf,.cg,.ch,.ci,.ck,.cl,.cm,.cn,.co,.cr,.cu,.cv,.cw,.cx,.cy,.cz,.de,.dj,.dk,.dm,.do,.dz,.ec,.ee,.eg,.eh,.er,.es,.et,.eu,.fi,.fj,.fk,.fm,.fo,.fr,.ga,.gd,.ge,.gf,.gg,.gh,.gi,.gl,.gm,.gn,.gp,.gq,.gr,.gs,.gt,.gu,.gw,.gy,.hk,.hm,.aq,.hn,.hr,.ht,.hu,.id,.ie,.il,.im,.in,.io,.iq,.ir,.is,.it,.je,.jm,.jo,.jp,.ke,.kg,.kh,.ki,.km,.kn,.kp,.kr,.kw,.ky,.kz,.la,.lb,.lc,.li,.lk,.lr,.ls,.lt,.lu,.lv,.ly,.ma,.mc,.md,.me,.mg,.mh,.mk,.ml,.mm,.mn,.mn,.mo,.mp,.mq,.mr,.ms,.mt,.mu,.mv,.mw,.mx,.my,.mz,.na,.nc,.ne,.nf,.ng,.ni,.nl,.no,.np,.nr,.nu,.nz,.om,.pa,.pe,.pf,.pg,.ph,.pk,.pl,.pm,.pn,.pr,.ps,.pt,.pw,.py,.qa,.re,.ro,.rs,.ru,.su,.рф,.rw,.sa,.sb,.sc,.sd,.se,.sg,.sh,.si,.sk,.sl,.sm,.sn,.so,.sr,.ss,.st,.bz,.su,.sv,.sx,.sy,.sz,.tc,.td,.tf,.tg,.th,.tj,.tk,.tl,.tp,.tm,.tn,.to,.tr,.tt,.tv,.tw,.tz,.ua,.ug,.uk,.us,.uy,.uz,.va,.vc,.ve,.vg,.vi,.vn,.vu,.wf,.ws,.ye,.yt,.za,.zm,.zw".split(",")]
besucht = []
verbindungen = {}

if os.path.exists("zustand_1000.json"):
    with open("zustand_1000.json", "r", encoding="utf-8") as f:
        objekt = json.load(f)
        zu_besuchend = objekt[0]
        besucht = objekt[1]
        verbindungen = objekt[2]
    print(zu_besuchend)
    print(len(zu_besuchend))

def finde_links(html, basis_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = set()

    for tag in soup.find_all('a', href=True):
        href = tag['href']
        if re.match(r"^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)(\.(?!$)|$)){4}$", href) == None: # Darf keine IP sein
            if href.startswith('http'):
                links.add(href)
            elif href.startswith('/'):
                links.add(basis_url.rstrip('/') + href)

    return links

def speichere():
    objekt = [zu_besuchend, besucht, verbindungen]
    with open('zustand_1000.json', 'w', encoding="utf-8") as f:
        json.dump(objekt, f)
    print(f'[INFO] Listen gespeichert: {len(besucht)} besuchte Seiten.')

keine_neuen = False
# Crawler-Schleife
while zu_besuchend:
    domain = zu_besuchend.pop(0)

    if domain in besucht:
        continue

    try:
        response = requests.get(domain, timeout=5)
        if response.status_code != 200:
            continue

        neue_links = finde_links(response.text, domain)

        neue_verbindungen = []
        for link in neue_links:
            parsed = urlparse(link)
            try:
                ip_last = int(parsed.netloc.split(".")[-1]) # URL ist IP, keine Domain
            except: # URL ist wahrscheinlich Domain
                if not ":" in parsed.netloc: # Schließen URLs mit Ports (in der Regel keine Domain-URLs) aus
                    neu_domain = f"http://{parsed.netloc}"

                    if neu_domain not in neue_verbindungen: # Falls die Domain nicht schon vorher auf dieser Domain-Seite gefunden wurde, wird sie als Verbindung hinzugefügt
                        neue_verbindungen.append(neu_domain)

                    if neu_domain not in besucht and neu_domain not in zu_besuchend and not keine_neuen: # Falls Domain-Verbindung neu gefunden, noch nicht zu zu_besuchend hinzugefügt, und falls wir noch sammeln, füge sie zu den zu Besuchenden hinzu
                        zu_besuchend.append(neu_domain)

        verbindungen[domain] = neue_verbindungen
        besucht.append(domain)
        print(f"[{len(besucht)}] Besuchte Seite: {domain}")

        # Alle 100 Seiten: Speichern
        if len(besucht) % 10 == 0:
            speichere()
        if len(besucht) == 1000:
            keine_neuen = True

        # Kleiner Delay, um Server zu schonen
        time.sleep(0.1)

    except Exception as e:
        print(f"[FEHLER] Konnte {domain} nicht laden: {e}")

# Zum Schluss: final speichern
speichere()