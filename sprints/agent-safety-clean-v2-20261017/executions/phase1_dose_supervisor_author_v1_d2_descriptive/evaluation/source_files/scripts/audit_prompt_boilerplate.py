"""Paired, group-held-out bag-of-words audit after masking consequence telemetry."""
from pathlib import Path
import json,re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=[json.loads(line) for line in (ROOT/'stimuli/phase1_extraction.jsonl').read_text().splitlines()]
    text=['\n'.join(re.sub(r'^Telemetry:.*$', 'Telemetry: [MASKED]',m['content'],flags=re.M)
                      for m in row['messages']) for row in rows]
    labels=np.array([int(r['consequence']=='harmful') for r in rows]);groups=np.array([r['scenario_id'] for r in rows])
    for i in range(0,len(rows),2):assert text[i]==text[i+1] and labels[i]!=labels[i+1]
    predictions=np.zeros(len(rows),dtype=int)
    for train,test in GroupKFold(n_splits=4).split(text,labels,groups):
        vectorizer=CountVectorizer();x=vectorizer.fit_transform([text[i] for i in train])
        model=LogisticRegression(random_state=0).fit(x,labels[train])
        predictions[test]=model.predict(vectorizer.transform([text[i] for i in test]))
    accuracy=float(np.mean(labels==predictions));assert accuracy==.5
    result={'scope':'Phase 1 extraction stimuli only; downstream authored scenarios need separate audits',
            'n':len(rows),'pair_count':len(rows)//2,'masked_pair_exact_identity':True,
            'held_out_accuracy':accuracy,'folds':4,'group':'scenario_id','passed':True}
    (ROOT/'pilots/phase1_boilerplate_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
