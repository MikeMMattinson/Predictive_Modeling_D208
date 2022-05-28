# helper functions

print('get_course_filename_str version: {}'.format('1.5'))
def get_course_filename_str(title: str, caption='',
             sect='', ftype = 'PNG',
            course = '', task = '',
               subfolder='figures',
               title_only=False) -> str:
    """
    Construct a filename for given figure or table
    Input:
      title:
      sect:
      caption:
      ftype:
      course:
      task:
      subfolder:
    """
    temp = subfolder + '/'  # subfolder for tables and figures, default is 'fig'
    if(title_only):
        temp += title 
    else:
        temp += course + '_'
        temp += task + '_'
        temp += subfolder[0:3] + " " + sect + '_' 
        temp += caption + '_'
        temp += title
    temp += '.' + ftype

    return temp.replace(' ','_').upper()


print('save_course_table_csv version: {}'.format('1.4'))
def save_course_table_csv(data, title: str, caption="", 
              sect='',course='', 
              task='', title_only=False):
    """
    Construct a filename for given figure or table
    Input:
      data:
      title:
      sect:
      caption:
      ftype:
      course:
      task:
      subfolder:
    """    
    # construct filename based on parameters
    f = get_course_filename_str(
        title=title, sect=sect, task=task, 
        caption=caption, ftype='CSV', 
        course=course, subfolder='tables', 
        title_only=title_only
    )
    data.to_csv(f, index=False, header=True) # create .csv file
    display(data.head(4).T) # display dataframe head data translated for vertical readibility
    print('shape: {}'.format(data.shape)) # describe dataframe rows and cols
    print('Table saved to: {}'.format(f)) # feedback to notebook   
    
    
print('describe_dataframe_type version: {}'.format('1.1'))      
def describe_dataframe_type(data):
    """
    Describe a set of data as Continuous or Categorical
    Input:
      data: dataframe to be described
    """ 
    for idx, c in enumerate(data.columns):
        if data.dtypes[c] in ('float', 'int', 'int64'):
            print('\n{}. {} is numerical (CONTINUOUS) - type: {}.'.format(idx+1, c, data.dtypes[c]))
            if data.dtypes[c] in ('int', 'int64'):
                numbers = data[c].to_numpy()
                print('  Unique: {}'.format(get_unique_values_list(numbers)))
            if data.dtypes[c] in ('float', 'float64'):
                print('  Min: {:.3f}  Max: {:.3f}  Std: {:.3f}'.format(data[c].min(), data[c].max(),data[c].std()))
            
        elif data.dtypes[c] == bool:
            print('\n{}. {} is boolean (BINARY): {}.'.format(idx+1,c,data[c].unique()))
        else:
            print('\n{}. {} is categorical (CATEGORICAL): {}.'.format(idx+1,c,data[c].unique()))  

    
print('create_scatter_plot_fig version: {}'.format('1.1'))
def create_scatter_plot_fig(data,feature,target,c,edgecolor,title,caption,course,task):
    """
    Create and save a custom scatter plot fiugre
    Input:
    data: dataframe
    feature:
    target:
    c:
    edgecolor:
    title:
    caption:
    course:
    """
    import matplotlib.pyplot as plt
    
    # define a couple of plot variables
    title = title + ' ' + str(feature) + ' ' + str(target)
    
    # create fig,ax
    fig,ax = plt.subplots()
    ax.scatter(data[feature],data[target],c=c,edgecolor=edgecolor)
    
    # set title
    ax.set_title(title.upper(), fontsize=16)
    
    # create filename and save
    f=getFilename(title=title, caption=caption,
                  course=course, task=task,
                  ftype='PNG', subfolder='figures')
    plt.gcf().text(0, -.05, f, fontsize=14)
    fig.savefig(f, dpi=150, bbox_inches='tight') 
    print('Figure saved to: {}'.format(f)) # feedback to notebook
    
    
print('create_barplot_num_vs_cat_fig version: {}'.format('1.9'))
def create_barplot_num_vs_cat_fig(data, num_target, cat_feature,
        title='',caption='',course='',task=''):
    """
    Create and save a custom bar plot fiugre
    Input:
    feature: feature (Categorical)
    target: target (Numerical)
    title:
    caption:
    course:
    """
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax=data.groupby(cat_feature).mean()[num_target].plot(kind='barh')
    title = 'Group ' + str(cat_feature) + ' by ' + str(num_target)
    ax.set_title(title.upper())
    ax.set_xlabel(('Ave. ' + num_target).upper())
    ax.set_ylabel('')
    f=get_course_filename_str(
        title=title, caption=caption,
        course=course, task=task, sect=sect,
        ftype='PNG', subfolder='figures')
    plt.gcf().text(0, -.05, f, fontsize=14)
    fig.savefig(f, dpi=150, bbox_inches='tight') 
    print('Figure saved to: {}'.format(f)) # feedback to notebook
    

print('create_distribution_plot_from_feature_fig version: {}'.format('1.9'))
def create_distribution_plot_from_feature_fig(
    data, cat_or_bool_feature, title='',
    caption='', course='', sect='', task=''):
    """
    This function creates a bar graph from pandas dataframe columns.
    Arguments:
        df: Pandas dataframe. Index will be x-axis. Categories and 
        associated amounts are from columns
        title: String. Name of the bar graph
    Outputs:
        Bar graph in console.
    """
    rate = data[cat_or_bool_feature].value_counts() / data.shape[0]
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax=rate.plot.barh(rot=0)
    for bars in ax.containers:
        ax.bar_label(bars)
    title = str(cat_or_bool_feature) + ' Distribution'
    ax.set_title(title.upper())
    f=get_course_filename_str(
        title=title, caption=caption,
        course=course, task=task, sect=sect,
        ftype='PNG', subfolder='figures')
    plt.gcf().text(0, -.05, f, fontsize=14)
    fig.savefig(f, dpi=150, bbox_inches='tight') 
    print('Figure saved to: {}'.format(f)) # feedback to notebook  
    

print('get_unique_values_list version: {}'.format('1.2'))    
def get_unique_values_list(numbers):
    """
    Input:
    numbers: array
    
    Ref: https://www.freecodecamp.org/news/python-unique-list-how-to-get-all-the-unique-values-in-a-list-or-array/
    """
    list_of_unique_numbers = []
    unique_numbers = set(numbers)
    for number in unique_numbers:
        list_of_unique_numbers.append(number)
    return list_of_unique_numbers


print('create_correlation_matrix version: {}'.format('1.2'))
def create_correlation_matrix(data,highest,title='',caption='',course='',task='',sect=''):
    """
    Create and save custom correlation matrix (heatmap) plot
    Input:
    data: dataframe of corr matrix
    c:
    edgecolor:
    title:
    caption:
    course:
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig,ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, fmt='.1f', 
        cmap='RdBu', center=0, ax=ax)
    ax.set_title(title.upper())
    fig.set_size_inches(8, 5)
    f=getFilename(title=title, caption=caption,
                  course=course, task=task, sect=sect,
                  ftype='PNG', subfolder='figures')
    plt.gcf().text(0, -.05, f, fontsize=14)
    plt.gcf().text(0, -.1, 'Top ' + str(highest) + ' Correlations:', fontsize=14, 
              horizontalalignment='left', verticalalignment='top') 
    plt.gcf().text(.04, -.15, get_top_n_correlations(data, n=highest).to_string(), 
              fontsize=14,horizontalalignment='left', verticalalignment='top') 
    fig.savefig(f, dpi=150, bbox_inches='tight') 
    print('Figure saved to: {}'.format(f)) # feedback to notebook
        

print('get_redundant_pairs version: {}'.format('1.0'))
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


print('get_top_n_correlations version: {}'.format('1.1'))
def get_top_n_correlations(df, n=5, threshhold=0.95):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]



print('create_simple_histogram_numerical_feature_fig version: {}'.format('1.10'))
def create_simple_histogram_numerical_feature_fig(
    data, numerical_feature, bins=5,
    title='', caption='', course='', sect='',
    task=''):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    #sns.distplot(df_clean['Tenure'], kde=True, color='red', bins=9)
    fig, ax = plt.subplots()
    sns.distplot(data[[numerical_feature]], bins,
                           color ='green', 
                              kde=True)
    #ax.hist(data, numerical_feature, bins=bins)  # df_clean[[f]].hist()

    plt.xlabel("Count")
    plt.ylabel(numerical_feature)

    # create and save figure image
    title = str(numerical_feature) +' Histogram'
    plt.title(title, fontsize=12)
    
    # add filename and save
    f=get_course_filename_str(title=title, caption=caption,
                  course=course, task='', sect=sect,
                  ftype='PNG', subfolder='figures')
    plt.gcf().text(0, -.05, f, fontsize=14)
    fig.savefig(f, dpi=150, bbox_inches='tight') 
    print('Figure saved to: {}'.format(f)) # feedback to notebook


print('create_stacked_barplot_cat_or_bool_feature_fig version: {}'.format('1.7'))
def create_stacked_barplot_cat_or_bool_feature_fig(
    data, cat_or_bool_feature, target,
    title='', caption='', course='', sect='',
    task='', bins=9, isCat=True):
    
    import matplotlib.pyplot as plt
    from pandas.core.frame import DataFrame
    y = DataFrame({"count": data.groupby([cat_or_bool_feature, target]).size()}).reset_index()

    x = y[cat_or_bool_feature].unique()

    fig, ax = plt.subplots()
    no = y[y[target] == False]
    yes = y[y[target] == True]

    ax.barh(x, no["count"], height=0.75, color="darkgreen", label="Churn No")
    ax.barh(x, yes["count"],height=0.75, color="lightgreen",label="Churn Yes", left=no["count"] )

    plt.xlabel("Count")
    plt.ylabel(cat_or_bool_feature)
    _, xmax = plt.xlim()
    plt.xlim(0, xmax + 300)

    # legend and grid
    ax.grid(True)
    ax.legend()

    # create and save figure image
    title = str(cat_or_bool_feature) + ' vs. ' + str(target) +' stacked'
    plt.title(title, fontsize=12)
    
    # add filename and save
    f=get_course_filename_str(title=title, caption=caption,
                  course=course, task='', sect=sect,
                  ftype='PNG', subfolder='figures')
    plt.gcf().text(0, -.05, f, fontsize=14)
    fig.savefig(f, dpi=150, bbox_inches='tight') 
    print('Figure saved to: {}'.format(f)) # feedback to notebook


print('create_stacked_histogram_num_feature_fig version: {}'.format('1.6'))
def create_stacked_histogram_num_feature_fig(
    data, numerical_feature, target,
    title='', caption='', course='', sect='',
    task='', bins=9, isCat=True):
    """
    Create and save a custom stacked histogram
    Input:
    data:
    feature: feature (Numerical)
    target: target (Yes/No)
    title:
    caption:
    course:
    task:
    bins:
    isCat: bool if target is cat, else num
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # create fig,ax
    fig,ax = plt.subplots()
    
    # define couple of plot variables
    if isCat==True:
        
        yes = data[data[target]=='yes'][numerical_feature]; yes_mean = yes.mean();
        no = data[data[target]=='no'][numerical_feature]; no_mean = no.mean()

    else:
        yes = data[data[target]==1][numerical_feature]; yes_mean = yes.mean();
        no = data[data[target]==0][numerical_feature]; no_mean = no.mean()   
        
    plt.hist([yes, no], bins=bins, stacked=True)
    title = str(numerical_feature) + ' ' + str(target)
    
    # add legend
    ax.legend([str(target)+'= Yes',str(target)+'= No'])
    
    # add datatable
    b = pd.cut(data[numerical_feature], bins=bins) # create bins (b) of numeric feature
    ct = pd.crosstab(data[target], b)
    plt.gcf().text(0.1, -.05, ct.T.to_string(), fontsize=14,
            horizontalalignment='left', verticalalignment='top')
    
    # set min-max x and y limits
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    
    # add title
    plt.title(title.upper(), fontsize=16)
    
    # axis labesl
    plt.xlabel(numerical_feature.upper())
    plt.ylabel(target.upper())
    
    # create group mean lines
    ax.axvline(yes_mean, color="blue", lw=2)  # yes mean
    ax.axvline(no_mean, color="orangered", lw=2)  # no mean
    
    # add filename and save
    f=get_course_filename_str(title=title, caption=caption,
                  course=course, task=task, sect=sect,
                  ftype='PNG', subfolder='figures')
    plt.gcf().text(0, -.05, f, fontsize=14)
    fig.savefig(f, dpi=150, bbox_inches='tight') 
    print('Figure saved to: {}'.format(f)) # feedback to notebook
    