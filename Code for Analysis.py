## Data Loading and Cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## Reading the Data

df = pd.read_csv('/Users/chinmaymalhotra/Desktop/Behavioural Decision Making/Spotify Final.csv')

df.head(5)
len(df)
## Add Customer Segment

def categorize_customer(row):
    if row['spotify_subscription_plan'] == 'Free (ad-supported)' and row['premium_sub_willingness'] == 'Yes':
        return 'T'
    elif row['spotify_subscription_plan'] == 'Free (ad-supported)' and row['premium_sub_willingness'] == 'No':
        return 'C'
    elif row['spotify_subscription_plan'] != 'Free (ad-supported)' and row['premium_sub_willingness'] == 'Yes':
        return 'S'
    elif row['spotify_subscription_plan'] != 'Free (ad-supported)' and row['premium_sub_willingness'] == 'No':
        return 'T'
    else:
        return 'Other'

df['Segment'] = df.apply(categorize_customer, axis=1)
## Checking for Null Values

df.isnull().sum()
## Replacing Null Values in String Columns as 'None'

df.fillna('None', inplace = True)
## Checking for Duplicated Values but not removing as user data and preferences can be duplicate in rare cases

df[df.duplicated(keep=False)]
## Checking Data Types Available

df.dtypes
## EDA
## Distribution of Age across Platform


plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['Age'], palette = "YlOrBr", edgecolor = 'black')
plt.title('Age Distribution', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Age Groups')
plt.ylabel('Frequency')
plt.show()

## Distribution of Usage Time Period

usage = df['spotify_usage_period'].value_counts()
explode = [0.05,0.05,0.05,0.05]
plt.figure(figsize=(10, 10), dpi=100)
plt.pie(usage, labels=usage.index, autopct='%1.1f%%', pctdistance=0.85, explode = explode, shadow=True, colors = sns.color_palette('YlOrBr'), wedgeprops=dict(width=0.3, linewidth = 1, edgecolor='black'))
plt.title('Usage Period Distribution', pad=20, fontsize=18, fontweight='bold', color='black')
plt.show()


## Distribution of Gender

plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['Gender'], palette = "YlOrBr", edgecolor = 'black')
plt.title('Gender Distribution', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Genders')
plt.ylabel('Frequency')
plt.show()

# Distribution of Device Used

df['spotify_listening_device'] = df['spotify_listening_device'].apply(lambda x: [device.strip() for device in x.split(',')])

devicedf = df.explode('spotify_listening_device')
device_order = devicedf['spotify_listening_device'].value_counts().index

plt.figure(figsize=(8, 8))
sns.countplot(data=devicedf, x='spotify_listening_device', order=device_order, palette = "YlOrBr", edgecolor = 'black')
plt.title('Distribution of Device Types Used', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Device Type')
plt.ylabel('Frequency')
plt.xticks(rotation=90)

plt.show()

## Distribution of Spotify Subscription Members and their Willingness to buy a Membership

plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['spotify_subscription_plan'], hue = df['premium_sub_willingness'], palette = "YlOrBr", edgecolor = 'black')
plt.title('Premium Subscription Members and Willingness', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Subscription Plan')
plt.ylabel('Frequency')
plt.show()

## Distribution of Spotify Subscription Willingness to Pay

plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['premium_sub_willingness'], palette = "YlOrBr", edgecolor = 'black')
plt.title('Distribution of Willingness to Pay for Premium Subscription', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Willingness')
plt.ylabel('Frequency')
plt.show()

## Distribution of Spotify Subscription Preferred Premium Plan

plantype = df['preffered_premium_plan'].value_counts().index

plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['preffered_premium_plan'], order = plantype, palette = "YlOrBr", edgecolor = 'black')
plt.title('Distribution of Preferred Premium Subscription Plan', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Plan Type')
plt.ylabel('Frequency')
plt.xticks(rotation = 90)
plt.show()

## Distribution of Preferred Listening Content Type

usage = df['preferred_listening_content'].value_counts()
explode = [0.07,0.05]
plt.figure(figsize=(10, 10), dpi=100)
plt.pie(usage, labels=usage.index, autopct='%1.1f%%', pctdistance=0.85, explode= explode, shadow=True, colors = sns.color_palette('YlOrBr'), wedgeprops=dict(width=0.3, linewidth = 1, edgecolor='black'))
plt.title('Preferred Listening Content Distribution', pad=20, fontsize=18, fontweight='bold', color='black')
plt.show()


## Distribution of Favourite Music Genre

plantype = df['fav_music_genre'].value_counts().index

plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['fav_music_genre'], order = plantype, palette = "YlOrBr", edgecolor = 'black')
plt.title('Distribution of Favourite Music Genre', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.xticks(rotation = 90)
plt.show()

## Distribution of Time Slot when User Likes to Listen to Songs

plantype = df['music_time_slot'].value_counts().index

plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['music_time_slot'], order = plantype, palette = "YlOrBr", edgecolor = 'black')
plt.title('Distribution of Preferred Time Slot to Listen', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Time of Day')
plt.ylabel('Frequency')
plt.show()

## Distribution of Influencial Mood for Listening

df['music_Influencial_mood'] = df['music_Influencial_mood'].apply(lambda x: [device.strip() for device in x.split(',')])

mooddf = df.explode('music_Influencial_mood')
mood_order = mooddf['music_Influencial_mood'].value_counts().index

plt.figure(figsize=(8, 8))
sns.countplot(data=mooddf, x='music_Influencial_mood', order=mood_order, palette = "YlOrBr", edgecolor = 'black')
plt.title('Distribution of Influential Moods for Users Preference of Music Choice', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Mood')
plt.ylabel('Frequency')
plt.xticks(rotation=90)

plt.show()

## Distribution of Listening Frequency for Users

df['music_lis_frequency'] = df['music_lis_frequency'].apply(lambda x: [activity.strip() for activity in x.split(',') if activity.strip() != ''])

freqdf = df.explode('music_lis_frequency')
freq_order = freqdf['music_lis_frequency'].value_counts().index

plt.figure(figsize=(8, 8))
sns.countplot(data=freqdf, x='music_lis_frequency', order=freq_order, palette = "YlOrBr", edgecolor = 'black')
plt.title('Distribution of Listening Frequency of Users', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Activities')
plt.ylabel('Frequency')
plt.xticks(rotation=90)

plt.show()

## Distribution of Music Exploration Methods

df['music_expl_method'] = df['music_expl_method'].apply(lambda x: [activity.strip() for activity in x.split(',') if activity.strip() != ''])

expldf = df.explode('music_expl_method')
expl_order= expldf['music_expl_method'].value_counts().index

plt.figure(figsize=(8, 8))
sns.countplot(data=expldf, x='music_expl_method', order=expl_order, palette = "YlOrBr", edgecolor = 'black')
plt.title('Distribution of Music Exploration Methods Used', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Method')
plt.ylabel('Frequency')
plt.xticks(rotation=90)

plt.show()

## Distribution of Reccomendation Engine Rating

fill_color = sns.color_palette("YlOrBr")[2]

plt.figure(figsize=(8, 8))
sns.kdeplot(df['music_recc_rating'], bw_adjust=3, fill=True, color=fill_color, edgecolor = 'brown')
plt.title('Distribution of Recommendation Engine Rating', pad=20, fontsize=18, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Density')

plt.show()


## Distribution of all Podcast Related Columns

columns_to_plot = ["pod_lis_frequency", "fav_pod_genre", "preffered_pod_format", "pod_host_preference", "preffered_pod_duration", "pod_variety_satisfaction"]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 18))
axes = axes.flatten()

for ax, column in zip(axes, columns_to_plot):
    order = df[column].value_counts(ascending=False).index  
    sns.countplot(data=df, x=column, ax=ax, order=order, palette="YlOrBr", edgecolor='black')
    ax.set_title(f'Distribution of {column.replace("_", " ").title()}',pad=20, fontsize=18, fontweight='bold', color='black')
    ax.set_xlabel(f'{column.replace("_", " ").title()}')
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()
## Distribution of Age vs.Gender

plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['Age'], hue = df['Gender'], palette = "YlOrBr", edgecolor = 'black')
plt.title('Age vs. Gender', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.show()

## Distribution of Age vs. Usage Period

plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['Age'], hue = df['spotify_usage_period'], palette = "YlOrBr", edgecolor = 'black')
plt.title('Age vs. Usage Period', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.show()

## Distribution of Usage Period vs. Reccomendation Rating 

usagerating = df.groupby('spotify_usage_period')['music_recc_rating'].mean()

plt.figure(figsize=(8, 8), dpi=100)
sns.lineplot(x = usagerating.index, y = usagerating.values , color = sns.color_palette("YlOrBr")[2])
plt.title('Usage Period vs. Average Reccomendation Rating ', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Usage Period')
plt.ylabel('Average Recommendation Rating')
plt.ylabel('Frequency')
# plt.ylim(0,5) ## Optional on what I want to show
plt.show()
## Distribution of Age vs. Favourite Genre

genre_by_age = pd.crosstab(df['Age'], df['fav_music_genre'])
genre_by_age = genre_by_age.div(genre_by_age.sum(axis=1), axis=0)
genre_by_age.plot(kind='bar', stacked=True, figsize=(10, 7), edgecolor='black')

plt.title('Age vs. Favourite Genre', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.legend(title='Favourite Music Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

## Distribution of Willingness to pay for Premium vs. Usage Period

plt.figure(figsize=(8, 8), dpi=100)
sns.countplot(x = df['spotify_usage_period'], hue = df['premium_sub_willingness'], palette = "YlOrBr", edgecolor = 'black')
plt.title('Usage Period vs. Willingness for Premium', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Usage Period')
plt.ylabel('Frequency')
plt.show()

## ## Distribution of Willingness to pay for Premium based on Usage Period and Current Plan

df['Usage_Period_And_Sub_Plan'] = df['spotify_usage_period'] + ' - ' + df['spotify_subscription_plan']

group_counts = df.groupby(['Usage_Period_And_Sub_Plan', 'premium_sub_willingness']).size().reset_index(name='Frequency')
plt.figure(figsize=(14, 8))
sns.barplot(data=group_counts, x='Usage_Period_And_Sub_Plan', y='Frequency',hue='premium_sub_willingness', palette="YlOrBr", edgecolor='black',
order=["Less than 6 months - Free (ad-supported)", "Less than 6 months - Premium (paid subscription)", "6 months to 1 year - Free (ad-supported)", "6 months to 1 year - Premium (paid subscription)",
"1 year to 2 years - Free (ad-supported)", "1 year to 2 years - Premium (paid subscription)", "More than 2 years - Free (ad-supported)", "More than 2 years - Premium (paid subscription)" ])


plt.title('Willingness to Pay for Premium by Usage Period and Subscription Plan', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Usage Period')
plt.ylabel('Frequency')
plt.legend(title='Willingness to Pay', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
## Distribution of Usage Period vs. Exploration Method

df['music_expl_method'] = df['music_expl_method'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
exploded_df1 = df.explode('music_expl_method')

plt.figure(figsize=(12, 8), dpi=100)
sns.countplot(x='spotify_usage_period', hue='music_expl_method', data=exploded_df1, palette="YlOrBr", edgecolor='black', 
order = ['Less than 6 months', '6 months to 1 year', '1 year to 2 years', 'More than 2 years'])

plt.title('Usage Period vs. Exploration Method', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Usage Period')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend(title='Exploration Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
## Distribution of Current Plan vs. Exploration Method

df['music_expl_method'] = df['music_expl_method'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
exploded_df2 = df.explode('music_expl_method')

plt.figure(figsize=(12, 8), dpi=100)
sns.countplot(x='spotify_subscription_plan', hue='music_expl_method', data=exploded_df2, palette="YlOrBr", edgecolor='black')

plt.title('Current Plan vs. Exploration Method', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Plan Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend(title='Exploration Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
## Distribution of Willingness vs. Exploration Method

df['music_expl_method'] = df['music_expl_method'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
exploded_df3 = df.explode('music_expl_method')

plt.figure(figsize=(12, 8), dpi=100)
sns.countplot(x='premium_sub_willingness', hue='music_expl_method', data=exploded_df3, palette="YlOrBr", edgecolor='black')

plt.title('Willingness to buy Premium vs. Exploration Method', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Willingness to Buy Premium')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend(title='Exploration Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
## Distribution of Willingness vs. Exploration Method

df['music_expl_method'] = df['music_expl_method'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
exploded_df4 = df.explode('music_expl_method')

explrating = exploded_df4.groupby('music_expl_method')['music_recc_rating'].mean()

plt.figure(figsize=(8, 8), dpi=100)
sns.lineplot(x = explrating.index, y = explrating.values , color = sns.color_palette("YlOrBr")[2])
plt.title('Exploration Method Average Reccomendation Rating Graph', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Exploration Method')
plt.ylabel('Average Recommendation Rating')
plt.ylabel('Frequency')
plt.xticks(rotation = 90)
# plt.ylim(0,5) ## Optional on what I want to show
plt.show()
## Checking for Statistically Significant Difference (T-Test)

from scipy import stats

df['music_expl_method_list'] = df['music_expl_method'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
df['uses_recommendations'] = df['music_expl_method_list'].apply(lambda methods: 'recommendations' in methods)
ratings_with_recommendations = df[df['uses_recommendations']]['music_recc_rating']
ratings_without_recommendations = df[~df['uses_recommendations']]['music_recc_rating']
t_stat, p_value = stats.ttest_ind(ratings_with_recommendations, ratings_without_recommendations, equal_var=False)

print(f"T-statistic: {t_stat}, P-value: {p_value}")

## Distribution of Willingness vs. Reccomendation Rating 

planrating = df.groupby('spotify_subscription_plan')['music_recc_rating'].mean()

plt.figure(figsize=(8, 8), dpi=100)
sns.lineplot(x = planrating.index, y = planrating.values , color = sns.color_palette("YlOrBr")[2])
plt.title('Current Plan vs. Average Reccomendation Rating ', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Plan Type')
plt.ylabel('Average Recommendation Rating')
plt.ylabel('Frequency')
# plt.ylim(0,5) ## Optional on what I want to show
plt.show()
## Checking for Statistically Significant Difference (T-Test)

ratings_with_plan = df[df['spotify_subscription_plan'] == 'Free (ad-supported)']['music_recc_rating']
ratings_without_plan = df[df['spotify_subscription_plan'] != 'Free (ad-supported)']['music_recc_rating']

# Perform the t-test
t_stat, p_value = stats.ttest_ind(ratings_with_plan, ratings_without_plan, equal_var=False)

print(f"T-statistic: {t_stat}, P-value: {p_value}")

## Distribution of Segment vs. Reccomendation Rating 

segmentrating = df.groupby('Segment')['music_recc_rating'].mean()

plt.figure(figsize=(8, 8), dpi=100)
sns.lineplot(x = segmentrating.index, y = segmentrating.values , color = sns.color_palette("YlOrBr")[2])
plt.title('Segment vs. Average Reccomendation Rating ', pad=20, fontsize=18, fontweight='bold', color='black')
plt.xlabel('Segment')
plt.ylabel('Average Recommendation Rating')
plt.ylabel('Frequency')
# plt.ylim(0,5) ## Optional on what I want to show
plt.show()
## Checking for Statistically Significant Difference (T-Test) (At Risk vs. Others)

ratings_with_T = df[df['Segment'] == 'T']['music_recc_rating']
ratings_without_T = df[df['Segment'] != 'T']['music_recc_rating']

# Perform the t-test
t_stat, p_value = stats.ttest_ind(ratings_with_T, ratings_without_T, equal_var=False)

print(f"T-statistic: {t_stat}, P-value: {p_value}")
## Data Pre-processing and Predictive Modelling
# Drop Unnecessary Columns

df.drop(columns = ['music_expl_method_list','uses_recommendations'], inplace = True)
df.drop(columns = ['premium_sub_willingness'], inplace = True)
## Ordinal Encoding for Usage Period

usage_period_mapping = {
    'Less than 6 months': 0,
    '6 months to 1 year': 1,
    '1 year to 2 years': 2,
    'More than 2 years': 3
}

df['spotify_usage_period'] = df['spotify_usage_period'].map(usage_period_mapping)

## One Hot Encoding all Categorical Vairbales after Flattening List like Values

list_columns = ['music_Influencial_mood', 'music_lis_frequency', 'music_expl_method', 'spotify_listening_device']

for column in list_columns:
    df[column] = df[column].apply(lambda x: x.strip("[]").split(", ") if isinstance(x, str) else x)
    df = df.explode(column)


categorical = ['Age', 'Usage_Period_And_Sub_Plan', 'spotify_subscription_plan', 'Gender', 'spotify_usage_period', 'spotify_listening_device',
       'preffered_premium_plan', 'preferred_listening_content',
       'fav_music_genre', 'music_time_slot', 'music_Influencial_mood',
       'music_lis_frequency', 'music_expl_method',
       'pod_lis_frequency', 'fav_pod_genre', 'preffered_pod_format',
       'pod_host_preference', 'preffered_pod_duration',
       'pod_variety_satisfaction']

df_encoded = pd.get_dummies(df, columns=categorical, drop_first = True)

df_encoded.head(5)

## Model Comparison using Repeated KFold

from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

label_encoder = LabelEncoder()
df_encoded['Segment_Encoded'] = label_encoder.fit_transform(df_encoded['Segment'])
X = df_encoded.drop(['Segment', 'Segment_Encoded'], axis=1)
y = df_encoded['Segment_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}
scoring_metrics = {
    'precision': make_scorer(precision_score, average='macro'),  # Use 'macro' average for multi-class classification
    'recall': make_scorer(recall_score, average='macro'),
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, average='macro')
}

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
results_list = []

for model_name, model in models.items():
    cv_results = cross_validate(model, X_train, y_train, cv=rkf, scoring=scoring_metrics, return_train_score=True, n_jobs=-1)
    for metric in scoring_metrics:
        train_mean = cv_results[f'train_{metric}'].mean()
        test_mean = cv_results[f'test_{metric}'].mean()
        train_std = cv_results[f'train_{metric}'].std()
        test_std = cv_results[f'test_{metric}'].std()

        results_list.append({
            "Model": model_name,
            "Metric": metric.capitalize(),
            "Train_Mean": train_mean,
            "Test_Mean": test_mean,
            "Train_Std": train_std,
            "Test_Std": test_std
        })

results_df = pd.DataFrame(results_list)
print(results_df)
# Hyperparameter Tuning using GridSearch ****  DO NOT RUN CONSIDERING THE TIME IT TAKES, BEST PARAMETERS HAVE ALREADY BEEN CALCULATED AND AVAILABLE IN THE NEXT LINE OF CODE ****

# xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# param_grid = {
#     'max_depth': [3, 5, 7],  
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'n_estimators': [100, 200, 300]
# }
# rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
#                            scoring='accuracy', cv=rkf, verbose=2, n_jobs=-1)

# grid_search.fit(X_train, y_train)
# print("Best parameters found: ", grid_search.best_params_)
# best_model = grid_search.best_estimator_
# predictions = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Test Accuracy: {accuracy}")
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming X_train, X_test, y_train, y_test have been defined previously
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.2,
    'max_depth': 7,
    'n_estimators': 300,
    'subsample': 0.6
}

# Initialize and fit the model with the best parameters
model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model using Repeated K-Fold CV
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_results = cross_val_score(model, X, y, cv=rkf, scoring='accuracy', n_jobs=-1)
print(f"Repeated K-Fold CV Mean Accuracy: {np.mean(cv_results):.4f}")
print(f"Repeated K-Fold CV Standard Deviation: {np.std(cv_results):.4f}")

# ROC Curve for each class
y_prob = model.predict_proba(X_test)
n_classes = len(np.unique(y))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting ROC curves
colors = ['blue', 'red', 'green', 'purple', 'orange']
plt.figure()
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.title('Learning curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc="best")
plt.show()

# Confusion Matrix
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
