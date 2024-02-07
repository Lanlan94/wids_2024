import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2):
    X = data.drop(columns=['DiagPeriodL90D'])
    y = data['DiagPeriodL90D']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

def fill_null_for_division_and_region(data, is_traing):
    if is_traing:

        df_region = data.groupby(['patient_zip3', 'Region', 'Division', 'patient_state']).agg(
            num = ('patient_age', 'count')
         ).reset_index()
    
        df_region.rename(columns = {"Region": 'Region_grouped',
                                    "Division": 'Division_grouped',
                                    "patient_state": 'patient_state_grouped'},
                                    inplace=True)

        df_region.sort_values(['patient_zip3', 'num'], ascending=False, inplace=True)

        #get mode for region and division for each zip3
        df_region_mode = df_region.groupby('patient_zip3').head(1)

        df_region_mode.to_csv('./data/processed/region_mode.csv', index=False)
    
    else:
        df_region_mode = pd.read_csv('./data/processed/region_mode.csv')

    data_joined = pd.merge(data, df_region_mode, on = 'patient_zip3', how='left')


    data_joined['Division'] = data_joined.apply(lambda x: x['Division_grouped'] if pd.isnull(x['Division']) else x['Division'], axis = 1)
    data_joined['Region'] = data_joined.apply(lambda x: x['Region_grouped'] if pd.isnull(x['Region']) else x['Region'], axis = 1)
    data_joined['patient_state'] = data_joined.apply(lambda x: x['patient_state_grouped'] if pd.isnull(x['patient_state']) else x['patient_state'], axis = 1)

    data_joined.drop(columns = ['Region_grouped', 'Division_grouped', 'patient_state_grouped', 'num'], inplace=True)

    return data_joined

def fill_null_for_other_regional_information(data, group_col, region_features, fill_strategy, is_traing):
    if is_traing:
        region_features_grouped = ['patient_zip3','Region','Division','patient_state'] + region_features
        df_zip_data = data.groupby(region_features_grouped, dropna=False).agg(
            num = ('patient_age', 'count')
            ).reset_index()
        
        df_zip_data_patient_state_avg = df_zip_data.groupby(group_col).agg(
            avg_population = ('population', fill_strategy),
            avg_density = ('density', fill_strategy),
            avg_age_median = ('age_median', fill_strategy),
            avg_age_under_10 = ('age_under_10', fill_strategy),
            avg_age_10_to_19 = ('age_10_to_19', fill_strategy),
            avg_age_20s = ('age_20s', fill_strategy),
            avg_age_30s = ('age_30s', fill_strategy),
            avg_age_40s = ('age_40s', fill_strategy),
            avg_age_50s = ('age_50s', fill_strategy),
            avg_age_60s = ('age_60s', fill_strategy),
            avg_age_70s = ('age_70s', fill_strategy),
            avg_age_over_80 = ('age_over_80', fill_strategy),
            avg_male = ('male', fill_strategy),
            avg_female = ('female', fill_strategy),
            avg_married = ('married', fill_strategy),
            avg_divorced = ('divorced', fill_strategy),
            avg_never_married = ('never_married', fill_strategy),
            avg_widowed = ('widowed', fill_strategy),
            avg_family_size = ('family_size', fill_strategy),
            avg_family_dual_income = ('family_dual_income', fill_strategy),
            avg_income_household_median = ('income_household_median', fill_strategy),
            avg_income_household_under_5 = ('income_household_under_5', fill_strategy),
            avg_income_household_5_to_10 = ('income_household_5_to_10', fill_strategy),
            avg_income_household_10_to_15 = ('income_household_10_to_15', fill_strategy),
            avg_income_household_15_to_20 = ('income_household_15_to_20', fill_strategy),
            avg_income_household_20_to_25 = ('income_household_20_to_25', fill_strategy),
            avg_income_household_25_to_35 = ('income_household_25_to_35', fill_strategy),
            avg_income_household_35_to_50 = ('income_household_35_to_50', fill_strategy),
            avg_income_household_50_to_75 = ('income_household_50_to_75', fill_strategy),
            avg_income_household_75_to_100 = ('income_household_75_to_100', fill_strategy),
            avg_income_household_100_to_150 = ('income_household_100_to_150', fill_strategy),
            avg_income_household_150_over = ('income_household_150_over', fill_strategy),
            avg_income_household_six_figure = ('income_household_six_figure', fill_strategy),
            avg_income_individual_median = ('income_individual_median', fill_strategy),
            avg_home_ownership = ('home_ownership', fill_strategy),
            avg_housing_units = ('housing_units', fill_strategy),
            avg_home_value = ('home_value', fill_strategy),
            avg_rent_median = ('rent_median', fill_strategy),
            avg_rent_burden = ('rent_burden', fill_strategy),
            avg_education_less_highschool = ('education_less_highschool', fill_strategy),
            avg_education_highschool = ('education_highschool', fill_strategy),
            avg_education_some_college = ('education_some_college', fill_strategy),
            avg_education_bachelors = ('education_bachelors', fill_strategy),
            avg_education_graduate = ('education_graduate', fill_strategy),
            avg_education_college_or_above = ('education_college_or_above', fill_strategy),
            avg_education_stem_degree = ('education_stem_degree', fill_strategy),
            avg_labor_force_participation = ('labor_force_participation', fill_strategy),
            avg_unemployment_rate = ('unemployment_rate', fill_strategy),
            avg_self_employed = ('self_employed', fill_strategy),
            avg_farmer = ('farmer', fill_strategy),
            avg_race_white = ('race_white', fill_strategy),
            avg_race_black = ('race_black', fill_strategy),
            avg_race_asian = ('race_asian', fill_strategy),
            avg_race_native = ('race_native', fill_strategy),
            avg_race_pacific = ('race_pacific', fill_strategy),
            avg_race_other = ('race_other', fill_strategy),
            avg_race_multiple = ('race_multiple', fill_strategy),
            avg_hispanic = ('hispanic', fill_strategy),
            avg_disabled = ('disabled', fill_strategy),
            avg_poverty = ('poverty', fill_strategy),
            avg_limited_english = ('limited_english', fill_strategy),
            avg_commute_time = ('commute_time', fill_strategy),
            avg_health_uninsured = ('health_uninsured', fill_strategy),
            avg_veteran = ('veteran', fill_strategy),
            avg_Ozone = ('Ozone', fill_strategy),
            avg_PM25 = ('PM25', fill_strategy),
            avg_N02 = ('N02', fill_strategy)
            ).reset_index()
        
        df_zip_data_patient_state_avg.to_csv('./data/processed/zip_data_patient_state_avg.csv', index=False)
    else:
        df_zip_data_patient_state_avg = pd.read_csv('./data/processed/zip_data_patient_state_avg.csv')
    
    data_joined = pd.merge(data, df_zip_data_patient_state_avg, on = group_col, how='left')

    for col in region_features:
        data_joined[col] = data_joined.apply(lambda x: x['avg_' + col] if pd.isnull(x[col]) else x[col], axis = 1)
        data_joined.drop(columns = ['avg_' + col], inplace=True)
    return data_joined

def fill_gender(data):
    data['patient_gender_cleaned'] = data.apply(lambda x: 'male' if x['breast_cancer_diagnosis_code'] in ('C50021', 'C50929') else x['patient_gender'], axis = 1 )
    return data

def fill_null_for_environmental(data, group_col, enviromental_features, fill_strategy, is_traing):
    if is_traing:
        features = ['patient_zip3','Region','Division','patient_state', 'Ozone', 'PM25', 'N02']

        df_division = data.groupby(features).agg(
            num = ('patient_age', 'count')
            ).reset_index()
        
        df_division_avg = data.groupby(group_col).agg(
            avg_Ozone = ('Ozone', fill_strategy),
            avg_PM25 = ('PM25', fill_strategy),
            avg_N02 = ('N02', fill_strategy)
        ).reset_index()

        df_division_avg.to_csv('./data/processed/zip_data_division_avg.csv', index=False)   
    else:
        df_division_avg = pd.read_csv('./data/processed/zip_data_division_avg.csv')

    data_joined = pd.merge(data, df_division_avg, on = group_col, how='left')

    for col in enviromental_features:
        data_joined[col] = data_joined.apply(lambda x: x['avg_' + col] if pd.isnull(x[col]) else x[col], axis = 1)
        data_joined.drop(columns = ['avg_' + col], inplace=True)
    return data_joined

def fill_null_for_payer_type(data, col, is_traing):
    if is_traing:
        df_race = data.groupby(['patient_race', col], dropna=False).agg(
            num=('patient_id', 'count')
            ).reset_index()
        
        df_race.rename(
            columns = {col: col+'_grouped'},
            inplace=True
        )

        df_race.sort_values(['patient_race', 'num'], ascending=False, inplace=True)

        #get mode for region and division for each zip3
        df_race_mode = df_race.groupby('patient_race', dropna=False).head(1)
    
        df_race_mode.to_csv('./data/processed/df_race_mode.csv', index=False)
    else:
        df_race_mode = pd.read_csv('./data/processed/df_race_mode.csv')

    data_joined = pd.merge(data, df_race_mode[['patient_race', col + '_grouped']], on = 'patient_race', how='left')

    data_joined[col] = data_joined.apply(lambda x: x[ col+ '_grouped'] if pd.isnull(x[col]) else x[col], axis = 1)
    
    data_joined.drop(columns = [ col + '_grouped'], inplace=True)
    return data_joined

def fill_null_for_bmi(data, col, fill_strategy, is_traing):
    if is_traing:
        # add age
        df_race = data.groupby(['patient_race'], dropna=False).agg(
            num=('patient_id', 'count'),
            avg_bmi = (col, fill_strategy)
            ).reset_index()
        
        df_race.to_csv('./data/processed/df_race.csv', index=False)
    
    else:
        df_race = pd.read_csv('./data/processed/df_race.csv')

    data_joined = pd.merge(data, df_race[['patient_race', 'avg_bmi']], on = 'patient_race', how='left')

    data_joined[col] = data_joined.apply(lambda x: x['avg_bmi'] if pd.isnull(x[col]) else x[col], axis = 1)
    
    data_joined.drop(columns = ['avg_bmi'], inplace=True)
    return data_joined

def fill_null_with_seperate_category(data, columns = []):
    for col in columns:
        data[col] = data[col].fillna('unknown')
    return data

def get_fill_for_na(data, columns=[], fill_strategy='separate_category'):
    
    filler_values = {}

    for column in columns:
        
        if fill_strategy == 'mean':
            fill_value = data[column].mean()   

        elif fill_strategy == 'median':
            fill_value = data[column].median()
                
        elif fill_strategy == 'mode':
            fill_value = data[column].mode()[0]
        
        elif fill_strategy == 'separate_category':
            fill_value = 'nan'

        filler_values[column] = fill_value

    return filler_values


def drop_cat(data):
    #this one is just to test how we do without any categorical features
    return data.select_dtypes(include=['number'])

def encode_cat(data,columns = []):
    return data

def normalize_data(data):
    return data

