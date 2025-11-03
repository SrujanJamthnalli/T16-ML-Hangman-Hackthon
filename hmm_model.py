import math, collections, numpy as np

ALPHABET = [chr(c) for c in range(ord('a'), ord('z') + 1)]
START, END, BLANK = "^", "$", "_"

def train_bigram_hmm(words, alpha=0.5):
    prev_symbols = [START] + ALPHABET
    next_symbols = ALPHABET + [END]
    counts = {p: collections.Counter() for p in prev_symbols}

    for w in words:
        s = [START] + list(w) + [END]
        for i in range(len(s)-1):
            counts[s[i]][s[i+1]] += 1

    trans_logp = {}
    for p in counts:
        trans_logp[p] = {}
        total = sum(counts[p].values()) + len(next_symbols)*alpha
        for n in next_symbols:
            prob = (counts[p][n] + alpha) / total
            trans_logp[p][n] = math.log(prob)
    return trans_logp

def emission_logp(hidden_state, observed, eps=1e-6):
    if observed == BLANK:
        return 0.0
    return 0.0 if hidden_state == observed else math.log(eps)

def forward_backward(masked, trans_logp):
    T = len(masked)
    fwd = np.full((T, 26), -np.inf)
    for j, s in enumerate(ALPHABET):
        fwd[0, j] = trans_logp[START][s] + emission_logp(s, masked[0])

    for t in range(1, T):
        for j, s in enumerate(ALPHABET):
            prevs = [fwd[t-1, k] + trans_logp[ALPHABET[k]][s] for k in range(26)]
            m = max(prevs)
            lse = m + math.log(sum(math.exp(v - m) for v in prevs))
            fwd[t, j] = lse + emission_logp(s, masked[t])

    bwd = np.full((T, 26), -np.inf)
    for j, s in enumerate(ALPHABET):
        bwd[T-1, j] = trans_logp[s][END]
    for t in reversed(range(T-1)):
        for j, s in enumerate(ALPHABET):
            nxts = [trans_logp[s][ALPHABET[k]] + emission_logp(ALPHABET[k], masked[t+1]) + bwd[t+1, k] for k in range(26)]
            m = max(nxts)
            lse = m + math.log(sum(math.exp(v - m) for v in nxts))
            bwd[t, j] = lse

    post = np.zeros((T, 26))
    for t in range(T):
        vals = [fwd[t, j] + bwd[t, j] for j in range(26)]
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        Z = sum(exps)
        post[t, :] = np.array(exps) / Z
    return post

def hmm_posterior_letter_scores(pattern, guessed, trans_logp):
    masked = [c if c != '_' else BLANK for c in pattern]
    post = forward_backward(masked, trans_logp)
    scores = {a: 0.0 for a in ALPHABET}
    for t, ch in enumerate(pattern):
        if ch == '_':
            for j, a in enumerate(ALPHABET):
                scores[a] += float(post[t, j])
    for a in guessed:
        scores[a] = 0.0
    s = sum(scores.values())
    if s > 0:
        for k in scores:
            scores[k] /= s
    return scores
