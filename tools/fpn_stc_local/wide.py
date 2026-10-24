"""One bounded wider pass over the SAME two shared calibration scalars."""
from . import calibration as cal

BASELINE=(1.,0.)
CANDIDATES=(BASELINE,)+tuple((g,b) for g in (.9,1.,1.1,1.25) for b in (-1.5,-1.,-.5,0.) if (g,b)!=BASELINE)
name=cal.name


def predict(body,provider,row,fw):
    prior=body.columns.prior_logits(fw['prior'])
    return cal.predict(fw['logits'],prior,fw['shape'],candidates=CANDIDATES)
