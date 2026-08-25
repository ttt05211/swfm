def harm_repair_counts(kta,wm,gt,support):
    m=support.bool(); kc=(kta==gt)&m; wc=(wm==gt)&m
    return {"repair":int((~kc&wc&m).sum()),"harm":int((kc&~wc&m).sum()),"preserve":int((kc&wc&m).sum()),"unresolved":int((~kc&~wc&m).sum()),"support":int(m.sum())}

def oracle_selector(kta,wm,gt,support):
    out=kta.clone(); choose=(wm==gt)&support.bool(); out[choose]=wm[choose]; return out
