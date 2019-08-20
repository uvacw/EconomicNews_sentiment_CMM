import os
import pandas
import numpy
import logging
import savReaderWriter
import scipy.stats
import datetime
from statsmodels import api
import readability_score

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def load_files(path_to_files):

    logger.info('Loading article files from {path_to_files}'.format(path_to_files=path_to_files))
    print_articles_1 = pandas.read_csv(os.path.join(path_to_files, 'newspaper1_febTOmay_AMCATdata_allesorigineel.csv'))
    print_articles_2 = pandas.read_csv('data/newspaper2_juneTOjuly_AMCATdata_allesorigineel.csv')
    web_articles     = pandas.read_csv(os.path.join(path_to_files, 'Internet_AllArticles-cleaned.csv'))

    logger.info('Loading annotations from {path_to_files}'.format(path_to_files=path_to_files))
    parser = savReaderWriter.SavReader('data/Inhoudsanalyse_AllesMerged_noICR_toneOnly.sav')
    decode = lambda x : x.decode()
    annotations = pandas.DataFrame([dict(zip(map(decode,parser.header),line)) for line in parser])

    logger.info('Merging files')
    print_rename_dict = {'id':'ID','headline':'title', 'medium':'source'}
    print_articles    = print_articles_1.append(print_articles_2).rename(columns=print_rename_dict)
    print_articles['type'] = 'print'
    web_rename_dict   = {'id':'ID', 'datum':'date'}
    web_articles      = web_articles.rename(columns=web_rename_dict)
    web_articles['type'] = 'web'

    columns_of_interest = ['ID','date','type','source','title','text']
    all_articles   = print_articles[columns_of_interest].append(web_articles[columns_of_interest])
    data = pandas.merge(annotations, all_articles, on='ID')

    # Drop missing title or body fields
    logger.info("Dropping items with missing body or title fields")
    nonzero_len = lambda x: len(str(x))>3
    data = data[data.title.map(nonzero_len) & data.text.map(nonzero_len)]

    # Lowercase text
    tolower = lambda x: x.lower()
    data['title'] = data.title.map(tolower)
    data['text']  = data.text.map(tolower)

    # Compute date index
    logger.info("Creating date index")
    todate = lambda d: datetime.datetime.strptime(d.replace('T',' ').split('.')[0],'%Y-%m-%d %H:%M:%S')
    data.index = data.date_y.map(todate)

    # Recode sources
    logger.info("Recoding sources and adding online/offline dummy")
    online_sources = ["telegraaf (www)", "nos (www)", "nrc (www)","volkskrant (www)","nu"]
    data['online'] = data.source.map(lambda x: x in online_sources)
    data['quality'] = data.source.replace({
        "Het Financieele Dagblad B.V." : 1,
        "De Telegraaf"                 : 0,
        "telegraaf (www)"              : 0,
        "NRC Media B.V."               : 1,
        "nrc (www)"                    : 1,
        "De Volkskrant"                : 1,
        "volkskrant (www)"             : 1,
        "Noordelijke Dagblad Combinatie / Dagblad van het Noorden" : 0,
        "Trouw"                        : 1,
        "Algemeen Dagblad"             : 0,
        "nos (www)"                    : 1,
        "Wegener NieuwsMedia BV"       : 0,
        "HDC Media B.V. / Noordhollands Dagblad" : 0,
        "nu"                           : 0,
        "Metro"                        : 0,
    })

    data['source'] = data.source.replace({
        "Het Financieele Dagblad B.V." : "FD",
        "De Telegraaf"                 : "Telegraaf",
        "telegraaf (www)"              : "Telegraaf",
        "NRC Media B.V."               : "NRC",
        "nrc (www)"                    : "NRC",
        "De Volkskrant"                : "Volkskrant",
        "volkskrant (www)"             : "Volkskrant",
        "Noordelijke Dagblad Combinatie / Dagblad van het Noorden" : "ND",
        "Trouw"                        : "Trouw",
        "Algemeen Dagblad"             : "AD",
        "nos (www)"                    : "NOS",
        "Wegener NieuwsMedia BV"       : "Wegener",
        "HDC Media B.V. / Noordhollands Dagblad" : "NHD",
        "nu"                           : "NU",
        "Metro"                        : "Metro"
    })


    logger.info("Loaded %s observations with %s columns" %data.shape)

    return data

def calculate_intercoder_reliability(data, article_id_field, coder_id_field, code_field, minimal_coder_overlap=2):
    from statsmodels.stats.inter_rater import fleiss_kappa

    # Test your sanity
    assert (data[article_id_field].value_counts()>1).sum() >0, Exception("No overlapping codes based on ''{article_id_field}' column' !".format(
        article_id_field = article_id_field
    ))

    # First, we construct an articles X categories matrix with counts for the number of coders
    grouped   = data[[article_id_field, coder_id_field, code_field]].groupby([article_id_field, code_field]).count()
    flattened = grouped.reset_index()
    pivot     = flattened.pivot(index=article_id_field, columns=code_field, values=coder_id_field)
    pivot_rec = pivot.replace({numpy.nan:0})
    useable   = pivot_rec[pivot_rec.sum(axis=1)>=minimal_coder_overlap]
    N         = len(useable)
    # Second, we compute Fleiss Kappa on the matrix
    kappa = fleiss_kappa(useable)
    return kappa, N

def recode_annotations(data):

    logger.info("Recoding title & body annotations")
    title_coding = {
        0  : -1,
        1  : 0,
        2  : 0,
        3  : 0,
        4  : 1,
        9  : numpy.nan,
        10 : numpy.nan
        }
    data['title_gold'] = data.Toon_Kop.replace(title_coding)

    text_coding = {
        0 : -2,
        1 : -1,
        2 : 0,
        3 : 0,
        4 : 1,
        5 : 2,
        9 : numpy.nan
        }
    data['text_gold'] = data.Posit_Nega.replace(text_coding)
    data = data[data.Economisch_Ja == 1]

    return data

def add_sentiments(data, field):
    from polyglot.text import Text
    from pattern.nl import sentiment
    from scripts.DANEW import DANEW
    from scripts.boukinator import Boukinator
    from resources.sentistrength.senti_client import multisent

    def polygloter(t):
        try:
            return Text(t, hint_language_code='NL').polarity
        except:
            return 0

    if not '%s_polyglot' %field in data.columns:
        logger.info('adding Polyglot')
        data['%s_polyglot' %field] = data[field].map(polygloter)

    if not '%s_pattern' %field in data.columns:
        logger.info('adding Pattern')
        patterner = lambda t: sentiment(t)[0]
        data['%s_pattern' %field] = data[field].map(patterner)

    if not '%s_DANEW' %field in data.columns:
        logger.info('adding DANEW')
        danew = DANEW()
        danewer = lambda t: danew.classify(t)['score']
        data['%s_DANEW' %field] = data[field].map(danewer)

    if not '%s_boukes' %field in data.columns:
        logger.info('adding Boukes et al')
        boukinator = Boukinator()
        boukinatorer = lambda t: boukinator.classify(t)['score']
        data['%s_boukes' %field] = data[field].map(boukinatorer)

    if not '%s_sentistrength' %field in data.columns:
        logger.info('adding Sentistrength')
        sentistrength = multisent(language='NL')
        data['%s_sentistrength' %field] = [int(s['positive']) + int(s['negative']) for s in sentistrength.run_batch(data[field])]

    if not '%s_recessie' %field in data.columns:
        logger.info('adding "recessie" classifier')
        # classify messages containing the word 'recessie' (EN: recession) as negative (-1)
        data['%s_recessie' %field ] = data[field].str.contains('recessie').map(int)*-1

    return data

def make_trinary(val, cutoff=0.5):
    if val<cutoff and val>-1*cutoff:
        return 0
    elif val<-1*cutoff:
        return -1
    elif val>cutoff:
        return 1
    else:
        return 0

def z_best(dataframe, field, ordered_asc_best):
    from scipy.stats.mstats import zscore
    nanzscore = lambda col: (col-col.mean())/col.std()
    for n in range(len(ordered_asc_best)):
        if n==len(ordered_asc_best): continue
        top_n = len(ordered_asc_best[n:])
        if top_n == 0: continue
        cols = ["%s_%s" %(field,el) for el in ordered_asc_best[n:]]
        scores = pandas.DataFrame()
        for col in cols:
            scores[col] = zscore(dataframe[col])
        if "%s_recessie" %field in scores:
            scores["%s_recessie" %field] = dataframe["%s_recessie" %field]
        means  = scores.apply(numpy.nanmean,1)
        dataframe["%s_top%s" %(field,top_n)] = means.tolist()
    return dataframe




def check_quality(data,field, trinairize=True,trinary_cutoff=0.5, beta=1, errors=False, normalization='zscore'):
    # Load support libraries
    import sklearn.metrics
    import sklearn.utils
    from scipy.stats.mstats import zscore


    # Define helper function to parse results
    def parse_PRFS_result(predicted,true,beta):
        result = {}

        output = sklearn.metrics.precision_recall_fscore_support(true,predicted, beta=beta)
        fbeta_label = "f{beta}".format(beta=beta)
        # Parse the output
        labels = sklearn.utils.multiclass.unique_labels(true,predicted)
        n_predicted = [sum(predicted==label) for label in labels]
        output = (*output,n_predicted)
        output_fields = ["precision","recall",fbeta_label,"support",'n_predicted']
        for output_field,row in zip(output_fields,output):
            result[output_field] = dict(zip(labels,row))

        result['precision'].update({'global': sklearn.metrics.precision_score(true,predicted, average='weighted')})
        result['recall'].update({'global': sklearn.metrics.recall_score(true,predicted, average='weighted')})
        result[fbeta_label].update({'global': sklearn.metrics.fbeta_score(true, predicted, beta=beta, average='weighted')})
        result['support'].update({'global':len(predicted[~predicted.isnull()])})

        return result

    if normalization=="min-max":
        # min-max scale metrics to push them into a [-1,1] interval, assuming a minimum upper-value bound of 1
        colnames = [name for name in data.columns if field+'_' in name and not '_err' in name]
        data = data[colnames] / data[colnames].abs().max().map(lambda x: max(x,1))

    # Select appropriate subset of data for quality metrics
    gold_field = "{field}_gold".format(field=field)
    data = data[~data[gold_field].isnull()]

    # Run the quality report function over each column
    if not errors:
        cols = [col for col in data.columns if field+"_" in col and not '_gold' in col and not "_err" in col]
    report = {}
    for col in cols:
        gold    = data[~data[col].isnull()&~data[gold_field].isnull()][gold_field]
        results = data[~data[col].isnull()&~data[gold_field].isnull()][col]

        # Trinarize if so required to cast
        # task as a three-class classification problem
        if trinairize:
            gold = pandas.Series(zscore(gold)).map(make_trinary)
            if normalization=='zscore':
                results = pandas.Series(zscore(results)).map(make_trinary).map(int)
            else:
                results = results.map(make_trinary).map(int)

        # (label , metric) : model : value
        for metric,label in parse_PRFS_result(results,gold,beta=beta).items():
            for label_label, label_value in label.items():
                label_label = str(label_label)
                if (label_label,metric) in report:
                    report[(label_label,metric)].update({col:numpy.round(label_value,2)})
                else:
                    report[(label_label,metric)] = {col:numpy.round(label_value,2)}


    return pandas.DataFrame(report)


def calculate_errors(data, field):

    cols = [col for col in data.columns if field+"_" in col and not '_gold' in col]
    gold = data['%s_gold' %field]
    for col in cols:
        data["{col}_err".format(col=col)] = abs(data[col]-gold)
    return data

def add_text_properties(data,field):
    from readability_score.calculators import flesch
    scores = []
    for t in data[field]:
        f = flesch.Flesch(t,'nl_NL')
        d = {k+"_%s" %field: v for k,v in f.scores.items()}
        d.update({'Flesch_%s' %field:f.reading_ease})
        scores.append(d)
    text_properties = pandas.DataFrame(scores)
    assert text_properties.shape[0] == data.shape[0], Exception("Input-Output length mismatch")
    text_properties.index = data.index
    return pandas.concat([data,text_properties], axis=1, join_axes=[data.index])



def analyze_errors(data, field, covars=True, standardize=True):
    import statsmodels.formula.api
    from collections import OrderedDict
    from scipy.stats.mstats import zscore
    gold = '%s_gold' %field
    methods = {colname.split('_')[1] for colname in data if "%s_" %field in colname and not "_gold" in colname}
    results = {}

    def sig_level(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""
    make_sig = lambda r,p: "%5.2f %-3.3s"%(r, sig_level(p))

    for method in methods:
        cols = [
                "{field}_{method}_err".format(field=field, method=method),
                "{field}_gold".format(field=field),
                "Flesch_{field}".format(field=field),
                "word_count_{field}".format(field=field),
                "quality",
                "online"
                ]
        testdata = data[cols].copy()
        testdata = testdata.dropna() # Statsmodels chokes in NaN values
        if standardize:
            cols = [col for col,dtype in zip(testdata.columns,testdata.dtypes) if dtype in ['int64','float64']]
            for col in cols:
                testdata[col] = zscore(testdata[col])
        form = "{field}_{method}_err ~ {field}_gold + quality + online + Flesch_{field} + word_count_{field}".format(field=field, method=method)
        model = statsmodels.formula.api.ols(form, data=testdata).fit()
        summary = OrderedDict()
        for par,b,p in zip(model.params.index, model.params, model.pvalues):
            summary[par] = make_sig(b,p)
        summary['N_observations'] = model.nobs
        summary['Model_F_score']  = make_sig(model.fvalue, model.f_pvalue)
        summary['Model_R2']       = numpy.round(model.rsquared,2)
        summary['Model_adj_R2']   = numpy.round(model.rsquared_adj,2)
        results.update({method: summary})
        fields = summary.keys()

    return pandas.DataFrame(results, index=fields)

def analyze_complexity(data, field, covars=True, standardize=True):
    import statsmodels.formula.api
    from collections import OrderedDict
    from scipy.stats.mstats import zscore
    gold = '%s_gold' %field
    methods = {colname.split('_')[1] for colname in data if "%s_" %field in colname and not "_gold" in colname}
    results = {}

    def sig_level(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""
    make_sig = lambda r,p: "%5.2f %-3.3s"%(r, sig_level(p))

    cols = [
            "{field}_gold".format(field=field),
            "Flesch_{field}".format(field=field),
            "word_count_{field}".format(field=field),
            "quality",
            "online"
            ]
    testdata = data[cols].copy()
    testdata = testdata.dropna() # Statsmodels chokes in NaN values
    if standardize:
        apply_zscore = lambda x: x.dtype in ['int64','float64'] and zscore(x) or x
        testdata = testdata.apply(apply_zscore, axis=1)
    form = "Flesch_{field} ~ {field}_gold + quality + online + word_count_{field}".format(field=field)
    model = statsmodels.formula.api.ols(form, data=testdata).fit()
    summary = OrderedDict()
    for par,b,p in zip(model.params.index, model.params, model.pvalues):
        summary[par] = make_sig(b,p)
    summary['N_observations'] = model.nobs
    summary['Model_F_score']  = make_sig(model.fvalue, model.f_pvalue)
    summary['Model_R2']       = numpy.round(model.rsquared,2)
    summary['Model_adj_R2']   = numpy.round(model.rsquared_adj,2)
    results.update({"Result": summary})
    fields = summary.keys()

    return pandas.DataFrame(results, index=fields)


def correlate_results(data, field, errors=False):
    if not errors:
        colnames = [name for name in data.columns if field+'_' in name and not '_err' in name]
    else:
        colnames = [name for name in data.columns if field+'_' in name and '_err' in name]
    return data[colnames].corr()

def correlation_tests(data, field, errors=False):
    import scipy.stats
    if errors:
        colnames = [name for name in data.columns if field+'_' in name and "_err" in name]
    else:
        colnames = [name for name in data.columns if field+'_' in name and not "_err" in name]

    def sig_level(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""
    make_sig = lambda r,p: "%4.2f %3.3s"%(r, sig_level(p))
    results = {col_x: { col_y: make_sig(*scipy.stats.pearsonr(data.dropna()[col_x],data.dropna()[col_y]))
                for col_y in colnames} for col_x in colnames}
    return pandas.DataFrame(results)

def cor_compare(ra, rb, na, nb):
    """Apply Fischers z transformation and calculate system

    Parameters
    ----
    ra : float (0,1)
        The first correlation coefficient
    rb : float (0,1)
        The second correlation coefficient
    na : int [0,inf]
        The sample size for the first correlation coefficient
    nb : int [0,inf]
        The sample size for the second correlation coefficient

    Returns
    ----
    OrderedDict
        ra : the first correlation coefficient
        rb : the second correlation coefficient
        na : the size of the first correlation coefficient sample
        nb : the size of the second correlation coefficient sample
        std_err : the pooled standard error
        z_score : the z-score for Fischer transformed r-value differences
        p_value : the p-value for significance difference in r-values
                  (H0: No difference between correlation coefficients)


    Notes
    ----
    adapted from http://vassarstats.net/rdiff.html
    """
    from collections import OrderedDict

    # Calculate fractions for z-score
    raplus  = 1*ra+1
    raminus = 1-ra
    rbplus  = 1*rb+1
    rbminus = 1-rb
    # Calculate z-scores
    za = (numpy.log(raplus)-numpy.log(raminus))/2
    zb = (numpy.log(rbplus)-numpy.log(rbminus))/2

    # Calculate pooled standard error
    se = numpy.sqrt((1/(na-3))+(1/(nb-3)) )

    # Calculate z-score for difference in Fischer's z scores
    z = (za-zb)/se

    # Convert z-score to pvalues
    z_abs = numpy.abs(z)
    pval  = (((((.000005383*z_abs+.0000488906)*z_abs+.0000380036)*z_abs+.0032776263)*z_abs+.0211410061*z_abs)+.049867347)*z_abs+1
    pval = pval ** -16
    pval = numpy.round(pval*10000)/10000

    # Create result dict

    result = OrderedDict(
        ra = ra,
        rb = rb,
        cordiff = ra - rb,
        na = na,
        nb = nb,
        std_err  = se,
        z_score  = z,
        p_value  = pval
    )

    return result

def mean_correlations(data,field):
    from collections import OrderedDict
    intervals = ["Article","1D","1W","1M"]
    results   = OrderedDict()
    n_observations = []
    for interval in intervals:
        if interval != "Article":
            aggregation = data.resample(interval).mean()
        else:
            aggregation = data.copy()
        n_observations.append(len(aggregation))
        cortest = correlation_tests(aggregation,field)
        results[interval] = {k:v for k,v in zip(cortest.index,cortest["%s_gold" %field])  if k!='%s_gold' %field }
    results_df = pandas.DataFrame(results)
    results_df.loc['N'] = n_observations
    return results_df

def get_fields(data, fieldpart):
    colnames = [name for name in data.columns if fieldpart+'_' in name]
    return data[colnames]

def compare_sentiment_means(data, field, covar=None, standardization="min-max"):
    from statsmodels.stats import anova
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from scipy.stats.mstats import zscore

    colnames = [name for name in data.columns if field+'_' in name and not '_err' in name]
    if standardization=="min-max":
        data_std = ((data[colnames] - data[colnames].min()) / (data[colnames].max() - data[colnames].min()))*2-1
    elif standardization == "z-score":
        data_std = data.copy()
        for col in colnames:
            data_std[col] = zscore(data[col])
    else:
        data_std = data
    if 'ID' in data.columns: # for normal data
        data_std['ID'] = data['ID']
    else: # for aggregated data, create placeholder IDs
        data_std['ID'] = [i for i in range(len(data_std))]
    data_std['%s_recessie' %field] = data['%s_recessie' %field]

    if covar: data_std[covar] = data[covar]

    if not covar:
        print(data_std.describe())
        melted = data_std.melt(id_vars=['ID'])
    else:
        print(data_std.groupby(covar).describe())
        melted = data_std.melt(id_vars=['ID',covar])

    melted = melted.dropna()

    if covar:
        formula = "value ~ variable*%s" %covar
    else:
        formula = "value ~ variable"
    aov   = anova.anova_single(ols(formula, melted).fit())
    print()
    print("## ANOVA results for differences in the mean of {field} sentiment".format(field=field))
    print()
    print(aov)

    if covar:
        melted['subgroups'] = ['%s_%s' %(col,covar) for col, covar in zip(melted['variable'],melted[covar])]
        posthoc = pairwise_tukeyhsd(melted['value'],melted['subgroups'])
    else:
        posthoc = pairwise_tukeyhsd(melted['value'], melted['variable'])

    print()
    print("## Tukey HSD post-hoc tests for mean differences in {field} sentiment".format(field=field))
    print()
    print(posthoc.summary())

    return data_std
