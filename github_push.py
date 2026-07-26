#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sube el LANGUAGE_ENGINE a un repo publico de GitHub.
USO: python3 github_push.py <usuario> <token> [nombre_repo]
Requiere el vault copiado a $HOME/engine_export/ (estructura limpia).
Crea el repo via API y sube todos los archivos (commit por commit o uno solo).
"""
import sys, os, json, base64, urllib.request, urllib.error

API="https://api.github.com"
SRC=os.path.expanduser("~/engine_export")
REPO=sys.argv[3] if len(sys.argv)>3 else "dscn-g-language-engine"

def req(method, url, token, data=None):
    headers={"Authorization":f"token {token}",
             "Accept":"application/vnd.github+json",
             "User-Agent":"dscn-push"}
    body=json.dumps(data).encode() if data is not None else None
    r=urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.read().decode(), resp.status
    except urllib.error.HTTPError as e:
        return e.read().decode(), e.code

def list_files(d):
    out=[]
    for root,_,files in os.walk(d):
        for f in files:
            if f.startswith("."): continue
            out.append(os.path.join(root,f))
    return out

def main():
    user,token=sys.argv[1],sys.argv[2]
    # crear repo
    _,st=req("POST",f"{API}/user/repos",token,
             {"name":REPO,"description":"DSCN-G Language Engine experiments",
              "public":True,"auto_init":False})
    print("create repo status:",st)
    # subir archivos
    files=list_files(SRC)
    # leer arbol existente (si repo tiene commit)
    for f in files:
        rel=os.path.relpath(f,SRC).replace(os.sep,"/")
        with open(f,"rb") as fh: content=fh.read()
        enc=base64.b64encode(content).decode()
        url=f"{API}/repos/{user}/{REPO}/contents/{rel}"
        _,st=req("PUT",url,token,{"message":f"add {rel}","content":enc})
        print(f"  {rel}: {st}")
    print("TERMINADO. Repo: https://github.com/%s/%s"%(user,REPO))

if __name__=="__main__":
    if len(sys.argv)<3:
        print("USO: python3 github_push.py <usuario> <token> [repo]"); sys.exit(1)
    main()
