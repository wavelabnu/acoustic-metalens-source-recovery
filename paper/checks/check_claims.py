"""Audit the adopted five-figure main and every retained Supplement claim.

Main numbers are read from TeX and recomputed from package/data drivers or the
explicitly frozen professor exports. The compact Supplement checks are retained
from the preceding audit. Removed main claims are not presented as current tests.
Missing numerical anchors remain errors, and tolerance follows printed precision.
"""
import functools
import json
import pathlib
import re
import sys
import numpy as np
_here=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(_here),str(_here.parent/'scripts'),str(_here.parents[1])]
from claims import ClaimNotFound,Ledger,Manuscript
from direct import Config,Lens,NOISE_ABS,analyze_T,lens_collective_modes,source_modes,stacked_matrix,transfer_matrix
from resonator_batch import BATCH
import figures as _fig
main,supp=Manuscript('main.tex'),Manuscript('supplement.tex')
led=Ledger()
_WORDS={w:i for i,w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
_WORDS['none']=0


def _supp_note(number):
    """Read one current note; missing prose remains an audit error."""
    match = re.search(r'\\subsection\*\{Supplementary Note '+str(number)+
        r'\.[^\n]*\}\s*(.*?)(?=\\subsection\*\{Supplementary Note|\\end\{document\})',
        supp.text, re.S)
    if match is None:
        raise ClaimNotFound(f'Supplementary Note {number} is absent')
    doc = Manuscript('supplement.tex')
    doc.text = match[1]
    # SI figures now precede the Notes. Locate their captions by stable labels,
    # not by accidental adjacency in the earlier two-column layout.
    labels={3:('fig:modes',),4:('fig:fsweep',),
            6:('fig:install','figS:qsweep'),9:('figS:qscatter',),
            10:('figS:budget',),11:('figS:exprecon',),15:('figS:singularpatterns',)}
    for block in re.findall(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',supp.text,re.S):
        if any(r'\label{'+label+'}' in block for label in labels.get(number,())):
            doc.text+='\n'+block
    doc.name += f' (Note {number})'
    return doc

def _note_words(doc, pattern):
    match = re.search(pattern.replace(' ', r'\s+'), doc.text)
    if match is None:
        raise ClaimNotFound(doc.name+': '+pattern)
    return [_WORDS[word.lower()] for word in match.groups()]

def _cosine_coordinate_checks():
    """Energy normalization of the retained cosine-to-channel maps."""
    from figures import cosine_svd_table
    led.header('Current Supplement: cosine decomposition')
    for group, arms in cosine_svd_table().items():
        for arm, C in arms.items():
            led.check_ints('complete cosine/SVD energy '+group+'/'+arm,
                [np.allclose(C.sum(axis=0),1), np.allclose(C.sum(axis=1),1), np.all(C>=0)],
                [True,True,True])

def T_of(kw, config=None, freqs=None):
    """Stacked operator, renormalized so sigma_1 sits DR_DB above the floor."""
    c = config or cfg
    T = stacked_matrix(c, FREQS if freqs is None else freqs, Lens(c),
                       src_resonant=True, **kw)
    return T * (SIG1 / np.linalg.svd(T, compute_uv=False)[0])

def svals(kw, **kwargs):
    return np.linalg.svd(T_of(kw, **kwargs), compute_uv=False)

def rank_of(s):
    return int(np.sum(s >= NOISE_ABS))

def finest_feature(kw, config=None):
    """Finest half-period (units of d) surviving the capture threshold."""
    c = config or cfg
    a = analyze_T(c, T_of(kw, config=c))
    return a['min_feature'] / c.d

def margin_of(overrides, kw, F=None):
    """Weakest-channel margin sigma_8/sigma_n at equal dynamic range."""
    c = type('CfgVar', (Config,), overrides)()
    freqs = c.freqs(cfg.f_max_count if F is None else F)
    T = stacked_matrix(c, freqs, Lens(c), src_resonant=True, **kw)
    s = np.linalg.svd(T, compute_uv=False)
    return (s[-1] / s[0]) * SIG1 / NOISE_ABS

def _aper(h):
    return margin_of({'lens_hwy_frac': cfg.src_sep_frac * h}, MONO)

def margin_phase(q, ph, F=None):
    F = cfg.f_max_count if F is None else F
    C2 = type('CfgVar', (Config,), {'q_factor': float(q)})()
    step = (C2.f_band_hi - C2.f_band_lo) / (F - 1)
    freqs = np.linspace(C2.f_band_lo, C2.f_band_hi, F) + ph * step
    sv = np.linalg.svd(stacked_matrix(C2, freqs, Lens(C2), **MONO), compute_uv=False)
    return (sv[-1] / sv[0]) * SIG1 / NOISE_ABS

def margin_median(q):
    return float(np.median([margin_phase(q, ph) for ph in COMB_PHASES]))

def first_q_below_floor():
    """Lowest integer Q at which the comb-phase median margin loses the channel.
    Same scan as figures.figS4_install."""
    for q in range(16, 301):
        if margin_median(float(q)) < 1.0:
            return q
    return None

def detuned_margin(fres):
    """Weakest-channel margin with the lens elements set to `fres`."""
    from robustness_checks import detune_T, rank_margin
    return rank_margin(detune_T(np.asarray(fres, float)))[1]

def single_freq_ranks(kw):
    out = []
    for f in sweep:
        T = transfer_matrix(cfg, f, lens, src_resonant=True, **kw)
        T = T * (SIG1 / np.linalg.svd(T, compute_uv=False)[0])
        out.append(rank_of(np.linalg.svd(T, compute_uv=False)))
    return np.array(out)

def meas_ranks(cut, tags=('nolens', 'withlens')):
    return [int(np.count_nonzero(MEAS[t] > cut)) for t in tags]

def _cost8(T):
    """Dynamic range at which the eighth channel clears the floor, dB."""
    s = np.linalg.svd(T, compute_uv=False)
    return 20 * np.log10(s[0] / s[-1])

def _rank64(T):
    s = np.linalg.svd(T, compute_uv=False)
    return rank_of(s * (SIG1 / s[0]))

def _margin(T):
    s = np.linalg.svd(T, compute_uv=False)
    return (s[-1] / s[0]) * SIG1 / NOISE_ABS

def _compact_model_protocol():
    from figures import driver_sweep, DRIVER_SWEEP
    from direct.physics import gpref
    d = _supp_note(1)
    led.check_ints('Supplement model element counts', [cfg.n_src,cfg.n_lens],
        _note_words(d,r'The (\w+) sources and (\w+) lens'))
    led.check_seq('Supplement lens damping', [cfg.q_factor,cfg.rw_loss],
        d.numbers(r'uses \$Q=NUM\$ and \$\\gamma=NUM\$'))
    led.check_seq('Supplement assumed source parameters',
        [cfg.f_res_src_frac,cfg.q_src,cfg.rw_src],
        d.numbers(r'\$f_\{0,\\rm src\}=NUMf_0\$, \$Q_\{\\rm src\}=NUM\$ and \$\\gamma_\{\\rm src\}=NUM\$'))
    led.check('Supplement resonant source amplitude fraction',100/(1+cfg.rw_src),
        d.number(r'1/\(1\+\\gamma_\{\\rm src\}\)=NUM\\%'))
    led.check('Supplement Green prefactor denominator',(2*np.pi)**2/gpref(1.),
        d.number(r'g_f=\(2\\pi f\)\^2/NUM'))
    led.check_seq('Supplement tone protocol',[cfg.f_max_count,cfg.f_band_lo,cfg.f_band_hi],
        d.numbers(r'\$F=NUM\$ tones over \$NUM\$--\$NUMf_0\$'))
    led.check('Supplement complex noise exponent',-np.log10(NOISE_ABS),
        d.number(r'\\sigma_n=10\^\{-NUM\}'))
    led.check_seq('Supplement normalized reference',[SIG1,DR_DB],
        d.numbers(r'\\sigma_1=NUM\$: the common relative budget is \$NUM\$'))
    rows=driver_sweep()
    led.check('Supplement sampled one-parameter driver cases',len(rows),
        d.number(r'\$NUM\$ sampled one-parameter cases'))
    for par,pat in (
        ('rw_src',r'absorption over \$NUM\$--\$NUM\$'),
        ('q_src',r'\$Q_\{\\rm src\}\$ over \$NUM\$--\$NUM\$'),
        ('f_res_src_frac',r'source resonance over \$NUM\$--\$NUMf_0\$')):
        led.check_seq('Supplement driver sweep '+par,
            [min(DRIVER_SWEEP[par]),max(DRIVER_SWEEP[par])],d.numbers(pat))
    led.check_ints('Supplement driver sweep channel counts',
        [min(r['lens'] for r in rows),min(r['nolens'] for r in rows),max(r['nolens'] for r in rows)],
        _note_words(d,r'retains (\w+) lens channels')+
        _note_words(d,r'counts range from (\w+) to (\w+)'))
    led.check('Supplement driver sweep finest pattern',max(r['lens_finest'] for r in rows),
        d.number(r'finest pattern, \$NUMd\$'))
    led.check_ints('Supplement states dimensionless model convention',
        ['so $f_0=c=1$' in d.text],[True])

def _compact_collective_modes():
    import rung_addressing as ra
    d=_supp_note(3)
    led.check_seq('Supplement collective resonance span',
        [modes['f_n'].min(),modes['f_n'].max()],d.numbers(r'span \$NUM\$--\$NUMf_0\$'))
    values=[ra.spectrum(cfg.q_factor,f) for f in ra.combs().values()]
    q=d.number(r'channel count is \$NUM\$')
    led.check_ints('Supplement frequency choices retain their count',[v[0] for v in values],
        [int(q.value)]*len(values))
    led.check_ints('Supplement number of frequency choices',[len(values)],
        _note_words(d,r'for (\w+) frequency choices'))
    led.check_seq('Supplement margins across frequency choices',
        [min(v[1] for v in values),max(v[1] for v in values)],
        d.numbers(r'margin .*?ranges from \$NUM\$ to \$NUM\\times\$'))
    qlo,qhi=d.numbers(r'tested \$Q=NUM\$--\$NUM\$')
    tuned=[ra.spectrum(q,ra.ladder_matched_comb(q)) for q in np.geomspace(qlo.value,qhi.value,12)]
    margin=d.number(r'margin near \$NUM\\times\$')
    led.check_ints('Supplement retuned high-Q channels and margin',
        [all(v[0]==cfg.n_src and margin.accepts(v[1]) for v in tuned)],[True])
    led.check_ints('Supplement retuned tone count',[len(ra.ladder_matched_comb(qlo.value))],
        _note_words(d,r'same (\w+) tones'))

def _compact_frequency_stack():
    d=_supp_note(4)
    led.check_ints('Supplement single-tone channel range',[min(seen),max(seen)],
        _note_words(d,r'shows (\w+) to (\w+) retained channels'))
    expected=d.numbers(r'give \$NUM\$, \$NUM\$, \$NUM\$, \$NUM\$ and \$NUM\$ channels')
    for name in ('independent + lens','coupled + lens'):
        got=[rank_of(svals(ARMS[name],freqs=cfg.freqs(f))) for f in range(1,6)]
        led.check_seq('Supplement first five tone counts '+name,got,expected)
    led.check_ints('Supplement no-lens stack counts',
        [rank_of(S['independent, no lens']),rank_of(S['coupled, no lens'])],
        _note_words(d,r'saturate at (\w+) for independent sources and (\w+) for coupled'))
    led.check_seq('Supplement single-tone figure sweep',[sweep[0],sweep[-1]],
        d.numbers(r'sweeps across \$NUM\$--\$NUM\\,f_0\$'))

def _compact_coupling_control():
    from robustness_checks import detune_T,rank_margin
    d=_supp_note(5)
    operators={(sc,couples):stacked_matrix(cfg,FREQS,lens,with_lens=True,
        src_scatters=sc,lens_couples=couples) for sc in (False,True) for couples in (True,False)}
    led.check_seq('Supplement coupling eighth-channel budgets',
        [_cost8(operators[sc,c]) for sc in (False,True) for c in (True,False)],
        d.numbers(r'rises from \$NUM\$ to \$NUM\$~dB for independent sources and from \$NUM\$ to \$NUM\$~dB'))
    led.check_seq('Supplement coupling retained counts',
        [_rank64(operators[sc,c]) for sc in (False,True) for c in (True,False)],
        d.numbers(r'counts fall from \$NUM\$ to \$NUM\$ and from \$NUM\$ to \$NUM\$'))
    led.check('Supplement coupling reference',DR_DB,d.number(r'At \$NUM\$~dB'))
    led.check_seq('Supplement selected measured resonances',
        [BATCH.selected_f0().min(),BATCH.selected_f0().max()],
        d.numbers(r'measured frequencies \$NUM\$--\$NUM\$~Hz'))
    led.check('Supplement batch frequency variation',100*BATCH.half_span_frac(),
        d.number(r'batch\x27s \$\\pmNUM\\%\$'))
    led.check('Supplement fractional lens linewidth',100*(1+cfg.rw_loss)/cfg.q_factor,
        d.number(r'\\simeqNUM\\%'))
    led.check_seq('Supplement detuned and identical margins',
        [detuned_margin(BATCH.selected_fres()),detuned_margin(np.ones(cfg.n_lens))],
        d.numbers(r'margin \$NUM\\times\$ versus \$NUM\\times\$'))
    led.check_ints('Supplement model counts with measured detuning',
        [rank_margin(detune_T(BATCH.selected_fres()))[0],rank_margin(detune_T(BATCH.selected_fres(),sever=True))[0]],
        _note_words(d,r'model lens elements retains (\w+) channels')+
        _note_words(d,r'frequencies retain (\w+) channels'))

def _compact_installation():
    d=_supp_note(6)
    led.check('Supplement finest-pattern decay length',1/kappa7/cfg.d,
        d.number(r'1/\\kappa_7=NUMd'))
    led.check('Supplement pitch in wavelengths',1/cfg.src_sep_frac,d.number(r'd=\\lambda/NUM'))
    m0,g0,m1,g1=d.numbers(r'from \$NUM\\times\$ at \$g=NUMd\$ to \$NUM\\times\$ at \$g=NUMd\$')
    led.check_seq('Supplement gap margins',
        [margin_of({'gap_frac':cfg.src_sep_frac*q.value},MONO) for q in (g0,g1)],[m0,m1])
    led.check('Supplement baseline gap',cfg.gap/cfg.d,g0)
    led.check_ints('Supplement common frequency shifts',[len(COMB_PHASES)],
        _note_words(d,r'tones and (\w+) common frequency shifts'))
    led.check('Supplement median high-Q crossing',Q_CROSS,d.number(r'near \$Q\\approxNUM\$'))
    lo,hi,half=d.numbers(r'shading \$NUM\$--\$NUM\\,d\$.*?dotted array half-height \$NUM\\,d\$')
    led.check('Supplement aperture caption array size',max(abs(cfg.y_src))/cfg.d,half)
    led.check_ints('Supplement aperture caption factor',
        [int(round(_aper(lo.value)/_aper(hi.value)))],_note_words(d,r'factor of about (\w+)'))
    led.check('Supplement installation caption noise level',DR_DB,d.number(r'at relative budget \$B=NUM\$~dB'))
    led.check('Supplement installation caption threshold',1.,d.number(r'level is \$NUM\$'))
    led.check_seq('Supplement Q-sweep caption budgets',[cfg.f_max_count,DR_DB,80,DR_DB-20*np.log10(2)],
        d.numbers(r'The \$F=NUM\$ curves use a \$NUM\$ dB relative budget; the \$F=NUM\$ curve uses \$NUM\$ dB'))
    led.check('Supplement dense-grid noise-ratio change',2.,
        d.number(r'noise-to-gain ratio by a factor of \$NUM\$'))
    led.check('Supplement Q-sweep caption crossing',Q_CROSS,d.number(r'crosses one near \$Q\\approxNUM\$'))

def _compact_capture_and_errors():
    import inspect
    from figures import pattern_support_table,measured_operator,MEAS_REL_FLOOR
    from direct import pattern_budget_sweep
    d=_supp_note(7)
    led.check('Supplement scalar tolerance',inspect.signature(pattern_budget_sweep).parameters['amplitude_rtol'].default,
        d.number(r'scalar amplitude estimation \$\\epsilon/g_m\\leNUM\$'))
    led.check('adopted main scalar amplitude tolerance',
        100*inspect.signature(pattern_budget_sweep).parameters['amplitude_rtol'].default,
        main.number(r'We use a \$NUM\\%\$ amplitude-error tolerance'))
    led.check('Supplement shape tolerance',inspect.signature(analyze_T).parameters['capture_frac'].default,
        d.number(r'shape capture uses \$C_m\\geNUM\$'))
    for group,arms in pattern_support_table().items():
        for arm,row in arms.items():
            cap=row['by_rank']
            led.check_ints('capture projector invariants '+group+'/'+arm,
                [np.allclose(cap.sum(axis=1),np.arange(9),atol=1e-12),
                 np.all(cap>=-1e-12) and np.all(cap<=1+1e-12),
                 np.all(np.diff(row['captures'],axis=0)>=-1e-12)],[True]*3)
    T=measured_operator('withlens');u,s,vh=np.linalg.svd(T,full_matrices=False)
    probes=source_modes(8);weights=abs(vh@probes.T)**2
    rank=np.count_nonzero(s>=MEAS_REL_FLOOR*s[0]);capture=weights[:rank].sum(axis=0)
    criteria=d.numbers(r'captured fractions \$NUM\$, \$NUM\$ and \$NUM\$')
    finest=[int(np.flatnonzero(capture>=q.value).max()) for q in criteria]
    led.check_seq('Supplement measured capture sensitivity',finest,
        d.numbers(r'finest passing indices \$NUM\$, \$NUM\$ and \$NUM\$'))
    q=d.number(r'sessions give \$NUM\$ throughout')
    for tag in ('nolens','nolens_b6'):
        _,sv,vv=np.linalg.svd(measured_operator(tag),full_matrices=False)
        cap=(abs(vv[sv>=MEAS_REL_FLOOR*sv[0]]@probes.T)**2).sum(axis=0)
        led.check_seq('Supplement no-lens capture sensitivity '+tag,
            [int(np.flatnonzero(cap>=c.value).max()) for c in criteria],[q]*len(criteria))
    led.check('Supplement m4 noiseless profile error',np.sqrt(1-capture[4]),
        d.number(r'relative profile error \$NUM\$'))
    led.check_seq('Supplement model finest shape half-periods',
        [FEAT[k] for k in ('independent, no lens','independent + lens','coupled, no lens','coupled + lens')],
        d.numbers(r'\$NUMd\$/\$NUMd\$ for independent sources and \$NUMd\$/\$NUMd\$ for coupled sources'))
    sigma=MEAS_REL_FLOOR*s[0]  # Hypothetical white noise for the algebra, not measured residuals.
    covariance=sigma**2*np.linalg.inv(T.conj().T@T)
    exact=np.real(np.einsum('mi,ij,mj->m',probes,covariance,probes))
    np.testing.assert_allclose(exact,sigma**2*np.sum(weights/s[:,None]**2,axis=0),rtol=1e-10)
    inverse=(vh[:rank].conj().T/s[:rank])@u[:,:rank].conj().T
    bias=inverse@T@probes.T-probes.T
    np.testing.assert_allclose(np.sum(abs(bias)**2,axis=0)+sigma**2*np.linalg.norm(inverse)**2,
        1-capture+sigma**2*np.sum(s[:rank]**-2),atol=1e-12)
    led.check_ints('Supplement uncertainty and TSVD risk identities',[True],[True])

def _compact_experimental_protocol():
    from figures import BENCH_D,measured_blocks
    d=_supp_note(8);data=measured_blocks('nolens');fr=data['freqs_hz']
    geometry=json.loads((_fig.EXP_RECON/'run_config_B.json').read_text(encoding='utf-8'))['physical_state']['gap_measurement_basis']
    led.check('Supplement mounted lens pitch mm',geometry['resonator_pitch_mm'],
        d.number(r'lens elements have \$NUM\$~mm pitch'))
    led.check_seq('Supplement original ten caliper gap readings mm',
        geometry['raw_resonator_gaps_mm_left_to_right'],
        d.numbers(r'were \$\(NUM,NUM,NUM,NUM,NUM,NUM,NUM,NUM,NUM,NUM\)\$~mm'))
    led.check('Supplement source pitch',1000*BENCH_D,d.number(r'have a \$NUM\$~mm pitch'))
    led.check_seq('Supplement acquired tones, band and drive setting',
        [len(fr),fr.min(),fr.max(),data['drive_level_v']],
        d.numbers(r'drive \$NUM\$ tones over \$NUM\$--\$NUM\$~Hz at \$NUM\$~V'))
    led.check('Supplement export block count',len(data['kinds']),d.number(r'contains \$NUM\$ blocks'))
    led.check_ints('Supplement apparatus counts',[data['D'].shape[2],cfg.n_lens,data['Y'].shape[2]],
        _note_words(d,r'(\w+) loudspeakers')+_note_words(d,r'with (\w+) selected')+
        _note_words(d,r'(\w+) microphones'))
    led.check_ints('Supplement calibration and acquisition repeats',
        [int(data['n_cal_reps']),len(_fig.measured_pattern_samples('staggered','nolens')[1])],
        _note_words(d,r'averaged over (\w+) repeats')+
        _note_words(d,r'and (\w+) separate acquisitions'))
    for tag in ('nolens','withlens'):
        led.check_ints('Supplement background recordings '+tag,
            [measured_blocks(tag)['Y_baseline_all'].shape[0]],
            _note_words(d,r'include (\w+) background recordings'))
    led.check_ints('Supplement repeated background and identical drives',
        [np.array_equal(measured_blocks('nolens_b6')['Y_baseline_all'],measured_blocks('withlens')['Y_baseline_all']),
         all(np.array_equal(data['D'],measured_blocks(t)['D']) for t in ('withlens','nolens_b6'))],[True,True])

def _compact_resonator_batch():
    d=_supp_note(9)
    led.check('Supplement resonator batch size',len(BATCH.ids),d.number(r'All NUM resonators'))
    led.check_seq('Supplement resonator frequency range and mean',
        [BATCH.f0_hz.min(),BATCH.f0_hz.max(),BATCH.f0_hz.mean()],
        d.numbers(r'frequencies span \$NUM\$--\$NUM\$~Hz, with mean \$NUM\$~Hz'))
    led.check('Supplement resonator figure batch count',len(BATCH.ids),
        d.number(r'Resonance frequencies of NUM elements'))
    led.check_ints('Supplement resonator figure mounted count',[len(BATCH.selected)],
        _note_words(d,r'of 21 elements; the (\w+) mounted'))

def _compact_measured_operator():
    from figures import BENCH_C,BENCH_D,BENCH_F0,measured_absolute_gains,audit_fraction
    from mic_survey import SURVEY,CSV
    d=_supp_note(10);tags=('nolens','withlens','nolens_b6')
    precision=float(re.search(r'to (\d\.\d) deg per microphone',CSV.read_text(encoding='utf-8'))[1])
    led.check('Supplement surveyed angular precision',precision,d.number(r'to \$\\pmNUM\^\{\\circ\}\$'))
    led.check_seq('Supplement surveyed geometry',[SURVEY.span_deg,SURVEY.arc_m,SURVEY.outlier[1]],
        d.numbers(r'uneven \$NUM\^\{\\circ\}\$ span\. The fitted arc radius is \$NUM\$~m; one microphone lies at \$NUM\$~m'))
    led.check_seq('Supplement far-field length and aperture',[2*(7*BENCH_D)**2/(BENCH_C/BENCH_F0),cfg.n_src-1],
        d.numbers(r'2D\^2/\\lambda_0=NUM\$~m and \$D=NUMd\$'))
    led.check_ints('Supplement microphones beyond ten Fraunhofer distances',
        [SURVEY.radius_m.min()>10*2*(7*BENCH_D)**2/(BENCH_C/BENCH_F0)],[True])
    led.check('Supplement separate calibration tests',len(PERR['nolens']['shots']),
        d.number(r'on \$NUM\$ separate recordings'))
    data=_fig.measured_blocks('nolens')
    led.check('Supplement recording sample count',np.prod(data['Y'].shape[1:]),
        d.number(r'all \$NUM\$ frequency--microphone samples'))
    led.check_seq('Supplement median recording prediction errors',
        [100*PERR[t]['median'] for t in tags],
        d.numbers(r'Median errors are \$NUM\\%\$, \$NUM\\%\$ and \$NUM\\%\$'))
    led.check_seq('Supplement recording-error middle range',
        [100*PERR[t][p] for t in ('nolens','withlens') for p in ('p10','p90')],
        d.numbers(r'spans \$NUM\\%\$--\$NUM\\%\$ without the lens and \$NUM\\%\$--\$NUM\\%\$ with it'))
    middle=d.number(r'middle \$NUM\\%\$')
    led.check_seq('Supplement share of recordings in the middle range',
        [100*np.mean((e['shots']>=e['p10'])&(e['shots']<=e['p90']))
         for e in (PERR['nolens'],PERR['withlens'])],[middle,middle])
    led.check('Supplement empirical relative cutoff',cut,d.number(r'cut \$NUM\\sigma_1\$'))
    led.check_seq('Supplement measured complex channel counts',meas_ranks(cut,tags),
        d.numbers(r'retains \$NUM\$, \$NUM\$ and \$NUM\$ complex channels'))
    led.check_seq('Supplement measured real channel counts',_real_ranks,
        d.numbers(r'giving \$NUM\$, \$NUM\$ and \$NUM\$ real directions'))
    lo,hi=d.numbers(r'cuts \$NUM\$--\$NUM\\sigma_1\$')
    led.check_ints('Supplement more lens channels throughout cutoff range',
        [all(meas_ranks(c)[1]>meas_ranks(c)[0] for c in np.linspace(lo.value,hi.value,61))],[True])
    led.check_ints('Supplement lens count at its own median',
        [_fig.measured_rank('withlens',PERR['withlens']['median'])],
        _note_words(d,r'median would retain (\w+) channels'))
    led.check_ints('Supplement common cut versus median errors',
        [min(PERR[t]['median'] for t in ('nolens','nolens_b6'))<cut<
         max(PERR[t]['median'] for t in ('nolens','nolens_b6'))],[True])
    g=measured_absolute_gains()
    led.check_seq('Supplement absolute error levels',
        [1000*g['level'][t] for t in tags],d.numbers(r'\$\(NUM,NUM,NUM\)\\times10\^\{-3\}\$'))
    led.check_seq('Supplement common absolute-cut counts',[g['count_common'][t] for t in tags],
        d.numbers(r'counts are \$NUM\$, \$NUM\$ and \$NUM\$'))
    fraction=audit_fraction('staggered','withlens')
    led.check_seq('Supplement staggered retained and omitted norm fractions',[100*fraction,100*(1-fraction)],
        d.numbers(r'retains \$NUM\\%\$ of its squared source norm and omits \$NUM\\%\$'))
    led.check_seq('Supplement staggered noiseless similarity and profile error',
        [np.sqrt(fraction),np.sqrt(1-fraction)],
        d.numbers(r'noise-free similarity is \$NUM\$ and relative profile error \$NUM\$'))
    led.check_seq('Supplement staggered physical scale',[BENCH_D*1000,BENCH_C/BENCH_F0/BENCH_D,BENCH_F0],
        d.numbers(r'd=NUM\$~mm is approximately \$\\lambda_0/NUM\$ at \$NUM\$~Hz'))

def _compact_recorded_recovery():
    from figures import measured_pattern_samples,measured_pattern_recon
    from scipy.stats import t
    d=_supp_note(11);names=('delta','pair_gap','pair_adj','staggered')
    led.check_ints('Supplement model display uses its noise-level cutoff, not main floor rank',
        [_rank64(stacked_matrix(cfg,FREQS,lens,with_lens=wl,src_resonant=True)) for wl in (False,True)],
        [int(q.value) for q in d.numbers(r'model retains \$NUM\$ channels without the lens and \$NUM\$ with it')])
    means=lambda name,tag,rank=None: float(measured_pattern_samples(name,tag,rank)[1].mean())
    led.check_ints('Supplement all displayed mean similarities improve',
        [all(means(n,'withlens')>means(n,'nolens') for n in names)],[True])
    led.check_seq('Supplement recorded staggered similarities',
        [means('staggered',t) for t in ('nolens','withlens')],
        d.numbers(r'similarity rises from \$NUM\$ to \$NUM\$'))
    confidence=d.number(r'its \$NUM\\%\$ Student-\$t\$')
    intervals=[]
    for tag in ('nolens','withlens'):
        samples=measured_pattern_samples('staggered',tag)[1]
        half=t.ppf(.5+confidence.value/200,len(samples)-1)*samples.std(ddof=1)/np.sqrt(len(samples))
        intervals.extend([samples.mean()-half,samples.mean()+half])
    led.check_seq('Supplement within-session Student-t intervals',intervals,
        d.numbers(r'intervals are \$\[NUM,NUM\]\$ and \$\[NUM,NUM\]\$'))
    sd=[measured_pattern_recon(n,'nolens')[2] for n in names]
    led.check_seq('Supplement no-lens sample standard deviations',[min(sd),max(sd)],
        d.numbers(r'deviations are \$NUM\$--\$NUM\$'))
    bound=d.number(r'below \$NUM\$ with it')
    led.check_ints('Supplement lens sample standard deviations',
        [all(measured_pattern_recon(n,'withlens')[2]<bound.value for n in names)],[True])
    led.check_seq('Supplement matched-count separated-pair exception',
        [means('pair_gap',tag,4) for tag in ('withlens','nolens')],
        d.numbers(r'lens similarity is \$NUM\$ compared with \$NUM\$ without the lens'))
    for rank in (4,5):
        led.check_ints('matched-rank recovery ordering '+str(rank),
            [all(means(n,'withlens',rank)>means(n,'nolens',rank) for n in ('delta','pair_adj','staggered')),
             means('pair_gap','withlens',rank)<means('pair_gap','nolens',rank)],[True,True])
    led.check('Supplement separated-pair distance',2*_fig.BENCH_D*1000,d.number(r'separated \$NUM\$~mm'))
    led.check('Supplement independent repeat staggered similarity',means('staggered','nolens_b6'),
        d.number(r'later no-lens measurement gives staggered similarity \$NUM\$'))
    drift=abs(means('staggered','nolens_b6')-means('staggered','nolens'))
    led.check_ints('Supplement inter-session shift exceeds within-session spread',
        [all(drift>measured_pattern_recon('staggered',tag)[2] for tag in ('nolens','nolens_b6'))],[True])

def _compact_prior_example():
    import ast
    from figures import prior_plane_example,prior_plane_neural,PRIOR_VIEW_CHANNELS,PRIOR_PLANE_OFFSET,PRIOR_PLANE_BUDGET
    d=_supp_note(14);row=prior_plane_example()
    prior_plane_neural(row)  # Validates the saved operator/prior fingerprint.
    saved=json.loads((_here.parents[1]/'notebooks/models/nn_prior_plane/results.json').read_text(encoding='utf-8'))
    led.check_seq('Supplement fixed coefficient patterns',np.array(PRIOR_VIEW_CHANNELS)+1,
        d.numbers(r'fixed lens-free real-stack patterns \$v_NUM,v_NUM\$'))
    led.check('Supplement coefficient-example budget',PRIOR_PLANE_BUDGET,d.number(r'At \$NUM\$~dB'))
    led.check_seq('Supplement coefficient-example truth',row['truth'],
        d.numbers(r'c_\\star=\(NUM,NUM\)\^\{\\mathsf T\}'))
    led.check_seq('Supplement fixed data displacement',PRIOR_PLANE_OFFSET,
        d.numbers(r'c_\\star\+L\(NUM,NUM\)\^\{\\mathsf T\}'))
    led.check_seq('Supplement prior standard deviations',np.sqrt(np.diag(row['priors']['correct']['covariance'])),
        d.numbers(r'C=\\operatorname\{diag\}\(NUM\^2,NUM\^2\)'))
    led.check_seq('Supplement coefficient prior means',
        [*row['priors']['correct']['mean'],*row['priors']['wrong']['mean']],
        d.numbers(r'means \$\(NUM,NUM\)\^\{\\mathsf T\}\$ or \$\(NUM,NUM\)\^\{\\mathsf T\}\$'))
    src=(_here.parents[1]/'notebooks/train_nn_prior_plane.py').read_text(encoding='utf-8')
    values={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(src).body
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
        and node.targets[0].id in ('STEPS','BATCH','SEEDS')}
    led.check_seq('Supplement coefficient-network steps and batch',[values['STEPS'],values['BATCH']],
        d.numbers(r'and NUM Adam optimization steps, batch NUM'))
    led.check_seq('Supplement coefficient-network seeds',values['SEEDS'],
        d.numbers(r'Seeds NUM, NUM and NUM'))
    led.check('Supplement retained reproduction seed',saved['figure_seed'],d.number(r'Seed NUM is retained'))
    validation=int(re.search(r'batch\((\d+), test_rng\)',src)[1])
    led.check('Supplement independent coefficient validation count',validation,
        d.number(r'each have NUM independent validation examples'))
    width=int(re.search(r'net = nn.Sequential\(nn.Linear\(2, (\d+)\)',src)[1])
    led.check('Supplement coefficient-network width',width,d.number(r'two-input, NUM-unit hidden-layer'))
    rates=[float(re.search(pattern,src)[1]) for pattern in (r'lr=([0-9.e-]+)',r'eta_min=([0-9.e-]+)')]
    led.check_seq('Supplement coefficient-network learning-rate exponents',
        [-np.log10(rate) for rate in rates],d.numbers(r'decay \$10\^\{-NUM\}\$--\$10\^\{-NUM\}\$'))
    led.check_ints('coefficient-example posterior and neural fingerprint reproduced',
        [all(np.allclose(saved['analytic'][k],row['estimates'][k]) for k in row['estimates'])],[True])
    from five_figure_claims import checkpoint_metrics
    reproduced=[]
    for key,items in saved['validation'].items():
        for item in items:
            metrics=checkpoint_metrics(_here.parents[1]/'notebooks/models/nn_prior_plane'/
                f'{key}_{item["seed"]}.pt',row['priors'][key],row['data_cov'],row['data_mean'])
            reproduced.append(metrics['geometry_sha256']==saved['geometry_sha256'] and all(
                np.allclose(metrics[k],item[k],rtol=2e-5,atol=2e-8)
                for k in ('estimate','heldout_rmse_vs_analytic','heldout_coefficient_mse')))
    led.check_ints('Supplement checkpoint weights reproduce the retained control statistics',reproduced,[True]*6)

def _compact_captions_and_priors():
    d=_supp_note(15)
    gallery=d.numbers(r'patterns \$v_NUM\$--\$v_NUM\$')
    led.check_seq('Supplement source-pattern gallery indices',[1,cfg.n_src],gallery)
    recon=_supp_note(11)
    data=_fig.measured_blocks('nolens');freq=data['freqs_hz']
    led.check_seq('Supplement recorded-reconstruction caption apparatus',
        [data['D'].shape[2],data['Y'].shape[2],len(freq),freq.min(),freq.max()],
        recon.numbers(r'\$NUM\$ loudspeakers, \$NUM\$ microphones, \$NUM\$ tones over \$NUM\$--\$NUM\$~Hz'))
    led.check('Supplement recorded-reconstruction caption budget',DR_DB,
        recon.number(r'at the \$NUM\$~dB reference budget'))

def _pattern_map_claims():
    """Complete matched-count sets in the compact Supplement, plus map invariants."""
    from figures import measured_operator
    doc = _supp_note(7)
    expected = [int(q.value) for q in doc.numbers(
        r'\$\\\{NUM,NUM,NUM,NUM\\\}\$ without the lens and \$\\\{NUM,NUM,NUM,NUM\\\}\$ with it')]
    for label, matrices in (
        ('coupled', [stacked_matrix(Config(), Config().freqs(20), Lens(Config()),
                     with_lens=wl, src_scatters=True) for wl in (False, True)]),
        ('independent', [stacked_matrix(Config(), Config().freqs(20), Lens(Config()),
                     with_lens=wl, src_scatters=False) for wl in (False, True)]),
        ('measured', [measured_operator(t) for t in ('nolens', 'withlens')])):
        sets = []
        for T in matrices:
            vh = np.linalg.svd(T, full_matrices=False)[2]
            weights = abs(vh @ source_modes(8).T)**2
            np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-12)
            sets.extend(np.flatnonzero(weights[:4].sum(axis=0) >= .5).tolist())
        led.check_ints('matched four-channel sets '+label, sets, expected)

cfg=Config();lens=Lens(cfg);FREQS=cfg.freqs(cfg.f_max_count);MODES=source_modes(cfg.n_src)
DR_DB=_supp_note(1).number(r'common relative budget is \$NUM\$').value
SIG1=NOISE_ABS*10**(DR_DB/20)
ARMS={'independent, no lens':dict(with_lens=False,src_scatters=False),
      'coupled, no lens':dict(with_lens=False,src_scatters=True),
      'independent + lens':dict(with_lens=True,src_scatters=False),
      'coupled + lens':dict(with_lens=True,src_scatters=True)}
S={k:svals(v) for k,v in ARMS.items()};FEAT={k:finest_feature(v) for k,v in ARMS.items()}
MONO=dict(with_lens=True,src_scatters=False)
COMB_PHASES=np.linspace(0,1,11)[:-1]
Q_CROSS=first_q_below_floor()
modes=lens_collective_modes(cfg,lens)
k_m7=np.pi*(cfg.n_src-1)/(cfg.n_src*cfg.d);kappa7=np.sqrt(k_m7*k_m7-cfg.k0(1.)**2)
_fs=re.search(r'fs_sweep = np\.linspace\(([\d.]+), ([\d.]+), (\d+)\)',(_here.parent/'scripts/figures.py').read_text(encoding='utf-8'))
if not _fs:raise ClaimNotFound('Supplement figure sweep missing')
sweep=np.linspace(float(_fs[1]),float(_fs[2]),int(_fs[3]))
seen=sorted({int(r) for kw in ARMS.values() for r in single_freq_ranks(kw)})
tags=('nolens','withlens','nolens_b6')
MEAS={}
for tag in tags:
    sv=np.linalg.svd(_fig.measured_operator(tag),compute_uv=False);MEAS[tag]=sv/sv[0]
cut=_supp_note(10).number(r'cut \$NUM\\sigma_1\$').value
PERR={tag:_fig.measured_prediction_error(tag) for tag in tags}
_real_ranks=[]
for tag in tags:
    H=_fig.measured_operator(tag);sv=np.linalg.svd(np.vstack([H.real,H.imag]),compute_uv=False)
    _real_ranks.append(int(np.count_nonzero(sv>cut*sv[0])))

def _supp_words(pattern, label):
    """The numbers a Supplement sentence writes in words, in order."""
    m = re.search(pattern.replace(' ', r'\s+'), supp.text)
    if m is None:
        raise ClaimNotFound(f'{label}: no match for {pattern}')
    return [_WORDS[g.lower()] for g in m.groups()]

def _test_pattern_checks():
    n, m0, m1 = supp.numbers(r'\\phi_m\(j\)\\propto\\cos\[\\pi m\(j\+\\tfrac12\)/NUM\]\$, '
                             r'\$m=NUM,\\ldots,NUM\$')
    j = np.arange(int(n.value))
    dct = np.array([np.cos(np.pi * m * (j + 0.5) / n.value) for m in range(int(m0.value),
                                                                            int(m1.value) + 1)])
    dct /= np.linalg.norm(dct, axis=1)[:, None]
    led.check_ints('Supp. Note 7: DCT-II probes match source_modes',
                   [int(n.value), int(m0.value), int(m1.value),
                    int(dct.shape == MODES.shape and np.max(np.abs(dct - MODES)) < 1e-12)],
                   [cfg.n_src, 0, cfg.n_src - 1, 1])
    two, n2, p = supp.numbers(r'\$\\cos\(NUM\\pi p j/NUM\)\$; at \$p=NUM\$ this is the staggered')
    rec = np.cos(two.value * np.pi * p.value * j / n2.value)
    alt = (-1.0) ** j
    drive = _fig.measured_driven_pattern('staggered')
    led.check_ints('Supp. Note 10: that recorded pattern is (-1)^j, as driven, and not DCT-II',
                   [int(n2.value), int(np.allclose(rec, alt)),
                    int(np.allclose(np.abs(drive @ alt) / (np.linalg.norm(drive) * np.sqrt(n2.value)), 1)),
                    int(np.max(np.abs(MODES @ alt)) / np.sqrt(n2.value) < 1 - 1e-6)],
                   [cfg.n_src, 1, 1, 1])

def _export_checks():
    import measured_vs_model as _mm                               # noqa: E402

    N = {t: _mm.noise_check(t) for t in ('A', 'B', 'B6')}
    s_lo, s_hi = supp.numbers(r'reduces the background to \$NUM\$--\$NUM\$ of a single recording')
    shrink = [N[t]['shrink'] for t in ('A', 'B', 'B6')]
    led.check('nine-shot average, smallest shrink', min(shrink), s_lo)
    led.check('nine-shot average, largest shrink', max(shrink), s_hi)
    shots = {N[t]['n'] for t in ('A', 'B', 'B6')}
    led.check_ints('Note 12: background shots averaged (in words)', sorted(shots),
                   _supp_words(r'Averaging (\w+) recordings', 'nine shots'))
    one, root_n = supp.numbers(r'\$NUM/NUM\$ expected for independent repeats')
    led.check_ints('Note 12: independent shots would shrink to 1/sqrt(n)',
                   [int(one.value), int(root_n.value)],
                   [1, int(round(1 / N['A']['ideal']))],
                   note=f"1/sqrt({N['A']['n']}) = {N['A']['ideal']:.4f}")

def _loaded_q_check():
    led.check('loaded model Q', cfg.q_factor / (1.0 + cfg.rw_loss),
              supp.number(r'loaded value is\s*\n?\$NUM\$'))

def _singular_pattern_gallery_checks():
    from figures import singular_pattern_table
    for measured in (False, True):
        tab = singular_pattern_table(measured=measured)
        for arm, row in tab.items():
            P, M, sv = row['patterns'], row['matrix'], row['singular_values']
            name = ('measured ' if measured else 'model ') + arm
            led.check_ints('SVD gallery orthonormal ' + name,
                          [int(np.allclose(P @ P.T, np.eye(8)))], [1])
            led.check_ints('SVD gallery gains and ordering ' + name,
                          [int(np.allclose(np.linalg.norm(M @ P.T, axis=0), sv)),
                           int(np.all(np.diff(sv) <= 0))], [1, 1])

from five_figure_claims import audit_main
audit_main(main,led)
led.guarded('Supplement cosine coordinates',_cosine_coordinate_checks)
led.guarded('Supplement matched-count sets',_pattern_map_claims)
led.guarded('_test_pattern_checks',_test_pattern_checks)
led.guarded('_export_checks',_export_checks)
led.guarded('_loaded_q_check',_loaded_q_check)
led.guarded('_singular_pattern_gallery_checks',_singular_pattern_gallery_checks)

led.header('Compact Supplement: every retained numerical statement')
for label,check in (
    ('model protocol and source assumptions',_compact_model_protocol),
    ('collective modes and tone selection',_compact_collective_modes),
    ('frequency stacking',_compact_frequency_stack),
    ('direct-coupling control',_compact_coupling_control),
    ('installation and unchanged figure captions',_compact_installation),
    ('capture criteria and uncertainty',_compact_capture_and_errors),
    ('experimental acquisition',_compact_experimental_protocol),
    ('resonator characterization',_compact_resonator_batch),
    ('measured operators and normalizations',_compact_measured_operator),
    ('recorded-source recovery and repeats',_compact_recorded_recovery),
    ('coefficient priors and network recipe',_compact_prior_example),
    ('retained figure captions',_compact_captions_and_priors)):
    led.guarded(label,check)

raise SystemExit(led.report())
