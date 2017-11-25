from scipy.io import arff
import pandas as pd
# Read file

data = pd.read_csv('german.data', header = None, delim_whitespace = True)
    
checking_status = list(data[0])
duration = list(data[1])
credit_history = list(data[2])
purpose = list(data[3])
credit_amount = list(data[4])
savings = list(data[5])
employment_since = list(data[6])
installment_rate = list(data[7]) # values are integers of {1, 2, 3, 4} --> no need to discretize
marital_sex = list(data[8])
other_debtors = list(data[9])
residence = list(data[10]) # values are integers of {1, 2, 3, 4} --> no need to discretize
property = list(data[11])
age = list(data[12]) # discretize age into 2 groups, young and old at age = 25
other_installment = list(data[13])
housing = list(data[14])
existing_credits = list(data[15]) # values are integers of {1, 2, 3, 4} --> no need to discretize
job = list(data[16])
dependents = list(data[17]) # values are integers of {1, 2} --> no need to discretize
telephone = list(data[18])
foreign_worker = list(data[19])
rating = list(data[20])

# Replace symbol by values using attribute discription in german.doc
symbol_description = {'A11' : '< 0 DM', 'A12' : '0 <= X < 200 DM)', 'A13' : '>= 200DM', 'A14' : 'no checking account',
                      'A30' : 'no credits taken/all credits paid', 'A31' : 'all credits paid at this bank', 'A32': 'existing credits paid back till now', 'A33' : 'delay paid in the past', 'A34' : 'critical account/other credits existing',
                      'A40' : 'car (new)', 'A41' : 'car (used)', 'A42' : 'furniture/equipment', 'A43' : 'radio/television', 'A44' : 'domestic appliances', 'A45' : 'repairs', 'A46' : 'education', 'A47' : 'vacation', 'A48' : 'retraining', 'A49' : 'business', 'A410' : 'others',
                      'A61' : '< 100 DM', 'A62' : '100 <= X < 500 DM)', 'A63' : '500 <= X < 1000 DM)', 'A64' : '>= 1000 DM', 'A65' : 'unknown or no savings',
                      'A71' : 'unemployed', 'A72' : '< 1 year', 'A73' : '1 <= X < 4 years)', 'A74' : '4 <= X < 7 years)', 'A75' : '>= 7 years',
                      'A91' : 'divorced male', 'A92' : 'divorced/married female', 'A93' : 'single male', 'A94' : 'married/widowed male', 'A95' : 'single female',
                      'A101' : 'none', 'A102' : 'co-applicant', 'A103' : 'guarantor',
                      'A121' : 'real estate', 'A122' : 'savings/life insurance', 'A123' : 'car/other', 'A124' : 'unknown/no property',
                      'A141' : 'bank', 'A142' : 'stores', 'A143' : 'none',
                      'A151' : 'rent', 'A152' : 'own', 'A153' : 'for free',
                      'A171' : 'unemployed/non-resident', 'A172' : 'unskilled/resident', 'A173' : 'skilled employee/official', 'A174' : 'management/self employed/highly qualified employee/officer',
                      'A191' : 'none', 'A192' : 'yes',
                      'A201' : 'yes', 'A202' : 'no'}

for i in range(0, len(data)):
    symbol = checking_status[i]
    checking_status[i] = symbol_description[symbol]
    symbol = credit_history[i]
    credit_history[i] = symbol_description[symbol]
    symbol = purpose[i]
    purpose[i] = symbol_description[symbol]
    symbol = savings[i]
    savings[i] = symbol_description[symbol]
    symbol = employment_since[i]
    employment_since[i] = symbol_description[symbol]
    symbol = marital_sex[i]
    marital_sex[i] = symbol_description[symbol]
    symbol = other_debtors[i]
    other_debtors[i] = symbol_description[symbol]
    symbol = property[i]
    property[i] = symbol_description[symbol]
    symbol = other_installment[i]
    other_installment[i] = symbol_description[symbol]
    symbol = housing[i]
    housing[i] = symbol_description[symbol]
    symbol = job[i]
    job[i] = symbol_description[symbol]
    symbol = telephone[i]
    telephone[i] = symbol_description[symbol]
    symbol = foreign_worker[i]
    foreign_worker[i] = symbol_description[symbol]
    symbol = rating[i]
    if symbol == 1:
        rating[i] = 'good'
    else:
        rating[i] = 'bad'
        
# discretize numerical attributes into 3 different bins
duration = list (pd.qcut(duration, 4, labels = ['3.999 < X <= 12 months', '12 < X <= 18 months', '18 < X <= 24 months', ' 24 < X <= 72 months']))  
credit_amount = list(pd.qcut(credit_amount, 4, labels = ['249.999 < X <= 1365.5', '1365.5 < X <= 2319.5', '2319.5 < X <= 3972.25', '3972.25 < X <= 18424.0']))
for i in range(0, len(age)):
    if age[i] <= 25:
        age[i] = 'young'
    else:
        age[i] = 'old'

# Summarize data
attributes_values = {'checking_status' : list(set(checking_status)),
                     'duration' : list(set(duration)),
                     'credit_history' : list(set(credit_history)),
                     'purpose' : list(set(purpose)),
                     'credit_amount' : list(set(credit_amount)),
                     'savings' : list(set(savings)),
                     'employment_since' : list(set(employment_since)),
                     'installment_rate' : list(set(installment_rate)),
                     'marital_gender' : list(set(marital_sex)),
                     'other_debtors' : list(set(other_debtors)),
                     'residence_since' : list(set(residence)),
                     'property' : list(set(property)),
                     'age' : list(set(age)),
                     'other_installments' : list(set(other_installment)),
                     'housing' : list(set(housing)),
                     'existing_credits' : list(set(existing_credits)),
                     'job' : list(set(job)),
                     'dependants' : list(set(dependents)),
                     'telephone' : list(set(telephone)),
                     'foreign_worker' : list(set(foreign_worker)),
                     'class' : list(set(rating))}

# Write to file
string = '@relation GermanCredit \n'
for attribute in attributes_values.keys():
    string = string + '@attribute ' + attribute + ' ' + '{'
    for i in range(0, len(attributes_values[attribute]) - 1):
        string = string + str(attributes_values[attribute][i]) + ', '
    string = string + str(attributes_values[attribute][-1]) + '}\n'

string = string + '@data \n'
for i in range(0, len(data)):
    string = string + checking_status[i] + ','\
                    + duration[i] + ','\
                    + credit_history[i] + ',' \
                    + purpose[i] + ','\
                    + credit_amount[i] + ','\
                    + savings[i] + ','\
                    + employment_since[i] + ','\
                    + str(installment_rate[i]) + ','\
                    + marital_sex[i] + ','\
                    + other_debtors[i] + ','\
                    + str(residence[i]) + ','\
                    + property[i] + ','\
                    + age[i] + ','\
                    + other_installment[i] + ','\
                    + housing[i] + ','\
                    + str(existing_credits[i]) + ','\
                    + job[i] + ','\
                    + str(dependents[i]) + ','\
                    + telephone[i] + ','\
                    + foreign_worker[i] + ','\
                    + rating[i] + '\n'

text_file = open('GermanCredit.arff', 'w')
text_file.write(string)
# test if arff file was correctly created

