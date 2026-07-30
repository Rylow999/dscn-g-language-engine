#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Push incremental: sube archivos nuevos/modificados al repo existente.
USO: python3 github_push_inc.py <usuario> <token> <ruta_local_rel> [ruta_local_rel...]
Ej: python3 github_push_inc.py Rylow999 TOKEN v0.15_sense v0.16_referencias README.md
"""
import sys, os, json, base64, urllib.request, urllib.error
API="https://api.github.com"
def req(method,url,token,data=None):
    h={"Authorization":f"token {token}","Accept":"application/vnd.github+json","User-Agent":"x"}
    b=json.dumps(data).encode() if data is not None else None
    r=urllib.request.Request(url,data=b,headers=h,method=method)
    try:
        with urllib.request.urlopen(r) as resp: return resp.read().decode(),resp.status
    except urllib.error.HTTPError as e: return e.read().decode(),e.code
def main():
    user,token=sys.argv[1],sys.argv[2]
    repo="dscn-g-language-engine"
    base=os.path.expanduser("~/engine_export")
    paths=sys.argv[3:]
    for p in paths:
        full=os.path.join(base,p)
        if os.path.isdir(full):
            for root,_,files in os.walk(full):
                for f in files:
                    if f.startswith("."): continue
                    fp=os.path.join(root,f)
                    rel=os.path.relpath(fp,base).replace(os.sep,"/")
                    _put(user,token,repo,rel,fp)
        else:
            rel=p.replace(os.sep,"/")
            _put(user,token,repo,rel,full)
    print("PUSH INCREMENTAL TERMINADO")
def _put(user,token,repo,rel,fp):
    with open(fp,"rb") as fh: content=base64.b64encode(fh.read()).decode()
    url=f"{API}/repos/{user}/{repo}/contents/{rel}"
    # si existe, necesita sha
    _,st=req("GET",url,token)
    data={"message":f"add/update {rel}","content":content}
    if st==200:
        try:
            cur=json.loads(_get(url,token)); data["sha"]=cur["sha"]
        except: pass
    _,s=req("PUT",url,token,data)
    print(f"  {rel}: {s}")
def _get(url,token):
    h={"Authorization":f"token {token}","User-Agent":"x"}
    return urllib.request.urlopen(urllib.request.Request(url,headers=h)).read().decode()
if __name__=="__main__":
    if len(sys.argv)<4: print("USO: github_push_inc.py <user> <token> <ruta...>"); sys.exit(1)
    main()
