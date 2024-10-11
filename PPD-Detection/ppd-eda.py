import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as pp
import seaborn as sns
import pandas.plotting
from IPython import display
import re
import mailbox
import csv

df = pd.read_csv("/kaggle/input/postpartum-depression/post natal data.csv")

df.head()

df.shape  
# (1503, 11)

df.columns  
# Index(['Timestamp', 'Age', 'Feeling sad or Tearful',
# 'Irritable towards baby & partner', 'Trouble sleeping at night',
# 'Problems concentrating or making decision',
# 'Overeating or loss of appetite', 'Feeling anxious', 'Feeling of guilt',
# 'Problems of bonding with baby', 'Suicide attempt'],
# dtype='object')

df.isnull().sum()
# Timestamp                                     0
# Age                                           0
# Feeling sad or Tearful                        0
# Irritable towards baby & partner              6
# Trouble sleeping at night                     0
# Problems concentrating or making decision    12
# Overeating or loss of appetite                0
# Feeling anxious                               0
# Feeling of guilt                              9
# Problems of bonding with baby                 0
# Suicide attempt                               0
# dtype: int64

'''
| Variable                                  | Key                               |
| ----------------------------------------- | --------------------------------- |
| Timestamp                                 | mm/dd/yyyy hh:mm                  |
| Age                                       | 25-30, 30-35, 35-40, 40-45, 45-50 |
| Feeling sad or tearful                    | yes, no, sometimes                |
| Irritable towards baby & partner          | yes, no, sometimes                |
| Trouble sleeping at night                 | yes, no, two or more days a week  |
| Problems concentrating or making decision | yes, no, often                    |
| Overeating or loss of appetite            | yes, no, not at all               |
| Feeling anxious                           | yes, no                           |
| Feeling of guilt                          | yes, no, maybe                    |
| Problems of bonding with baby             | yes, no, sometimes                |
| Suicide attempt                           | yes, no, not interested to say    |
'''

df = df.dropna()

df.isna().sum().sum()
# 0

df.rename(columns = {'Feeling sad or Tearful':'Sad_Tearful',
    'Irritable towards baby & partner':'Irritable', 'Trouble sleeping at night': 'Trouble_Sleeping',
    'Problems concentrating or making decision':'Problems_Focusing',
    'Overeating or loss of appetite': 'Eating_Disorder', 'Feeling anxious': 'Anxious', 'Feeling of guilt':'Guilt', 
    'Problems of bonding with baby':'Problems_Bonding','Suicide attempt':'Suicide_Attempt'},inplace = True)

df.columns
# Index(['Timestamp', 'Age', 'Sad_Tearful', 'Irritable', 'Trouble_Sleeping',
#     'Problems_Focusing', 'Eating_Disorder', 'Anxious', 'Guilt',
#     'Problems_Bonding', 'Suicide_Attempt'],
#     dtype='object')

pd.DataFrame(df.Timestamp.value_counts())

pd.DataFrame(df.Sad_Tearful.value_counts())

sad_tearful = df.groupby("Sad_Tearful").Age.value_counts(normalize=True)

sad_tearful.unstack()

irritable = df.groupby("Irritable").Age.value_counts(normalize=True)

irritable.unstack()

pd.DataFrame(df.Trouble_Sleeping.value_counts())

trouble_sleeping = df.groupby("Trouble_Sleeping").Age.value_counts(normalize=True)

trouble_sleeping.unstack()

pd.DataFrame(df.Problems_Focusing.value_counts())

problems_focusing = df.groupby("Problems_Focusing").Age.value_counts(normalize=True)

problems_focusing.unstack()

pd.DataFrame(df.Eating_Disorder.value_counts())

eating_disorder = df.groupby("Eating_Disorder").Age.value_counts(normalize=True)

eating_disorder.unstack()

pd.DataFrame(df.Anxious.value_counts())

anxious = df.groupby("Anxious").Age.value_counts(normalize=True)

anxious.unstack()

pd.DataFrame(df.Guilt.value_counts())

guilt = df.groupby("Guilt").Age.value_counts(normalize=True)

guilt.unstack()

pd.DataFrame(df.Problems_Bonding.value_counts())

problems_bonding = df.groupby("Problems_Bonding").Age.value_counts(normalize=True)

problems_bonding.unstack()

pd.DataFrame(df.Suicide_Attempt.value_counts())

suicide_attempt = df.groupby("Suicide_Attempt").Age.value_counts(normalize=True)

suicide_attempt.unstack()

pp.figure(figsize=(10,6))
pp.subplot(1,2,1);df.Age.value_counts().plot(kind='pie', colors=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Mother Age Groups')
pp.subplot(1,2,2);df.Age.value_counts().plot(kind='bar', color=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Mother Age Groups')
# Text(0.5, 1.0, 'Mother Age Groups')

pp.figure(figsize=(10,6))
pp.subplot(1,2,1); sad_tearful.plot(kind='barh', color = '#fdbe85'); pp.title('Feeling Sad or Tearful ')
pp.subplot(1,2,2);df.Sad_Tearful.value_counts().plot(kind='pie', colors=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Feeling Sad or Tearful ')
pp.subplot(1,3,2);df.Sad_Tearful.value_counts().plot(kind='bar', color=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Feeling Sad or Tearful')
pp.subplot(1,2,2); sad_tearful.plot(kind='barh', color = '#a63603')
# Text(0.5, 1.0, 'Feeling Sad or Tearful ')

pp.figure(figsize=(10,6))
pp.subplot(1,3,1);df.Irritable.value_counts().plot(kind='bar', color=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Irritable Towards Baby & Partner ')
pp.subplot(1,3,2); irritable.plot(kind='bar', color = '#fdbe85'); pp.title('Irritable Towards Baby & Partner ')
# Text(0.5, 1.0, 'Irritable Towards Baby & Partner ')

plt.figure(figsize = (10,5))
df.Trouble_Sleeping.value_counts().plot(kind='pie', colors=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Trouble Sleeping at Night')
trouble_sleeping.unstack().plot(kind='barh',color=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Trouble Sleeping at Night')
# Text(0.5, 1.0, 'Trouble Sleeping at Night')

plt.figure(figsize = (10,5))
df.Eating_Disorder.value_counts().plot(kind='bar', color=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Overeating or Loss of Appetite')
eating_disorder.unstack().plot(kind='barh',color=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Overeating or Loss of Appetite ')
# Text(0.5, 1.0, 'Overeating or Loss of Appetite')

plt.figure(figsize = (10,5))
df.Problems_Focusing.value_counts().plot(kind='pie', colors=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Problems Concentrating or Making Decisions')
plt.figure(figsize=(10,6))
problems_focusing.unstack().plot(kind='barh',color=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Problems Concentrating or Making Decisions ')
# Text(0.5, 1.0, 'Problems Concentrating or Making Decisions')

plt.figure(figsize = (10,5))
df.Guilt.value_counts().plot(kind='pie', colors=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Feeling of Guilt ')
guilt.unstack().plot(kind='barh',color=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Feeling of Guilt ')
# Text(0.5, 1.0, 'Feeling of Guilt')

pp.figure(figsize=(10,6))
pp.subplot(1,2,1); problems_bonding.plot(kind='barh', color = '#fdbe85'); pp.title('Problems of Bonding with Baby ')
pp.subplot(1,2,2);df.Problems_Bonding.value_counts().plot(kind='pie', colors=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Problems of Bonding with Baby ')
# Text(0.5, 1.0, 'Problems of Bonding with Baby ')

plt.figure(figsize = (10,5))
df.Suicide_Attempt.value_counts().plot(kind='pie', colors=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Suicide Attempt  ')
suicide_attempt.unstack().plot(kind='barh',color=['#feedde', '#fdbe85','#fd8d3c','#e6550d','#a63603']); pp.title('Suicide Attempt  ')
# Text(0.5, 1.0, 'Suicide Attempt  ')

'''
The study analyzed data from 1,491 female participants aged between 25 and 50, with a noticeable concentration of responses in the 40-45 age range. It revealed significant findings related to postpartum mental health issues, suggesting a prevalence of symptoms associated with Postpartum Depression (PPD) and Postpartum Psychosis (PPP) much higher than previously documented. Notably, two-thirds of the participants reported experiencing sadness or tearfulness, a common symptom of PPD. The distribution of other symptoms such as irritability, sleep disturbances, concentration problems, guilt, bonding issues, and suicidal tendencies further indicates a concerning level of postpartum mental health challenges among the respondents.

The data suggests that the incidence of PPD might be higher than the established statistic of one in nine new mothers, based on the substantial number of participants reporting key symptoms. Furthermore, the responses related to suicide attempts point towards a potential underestimation of PPP, which is traditionally thought to affect 4 in 1,000 new mothers. The findings underscore the urgent need for enhanced awareness and proactive mental health support for expecting and new mothers.

To address these challenges, it is recommended that prenatal education includes comprehensive information on PPD and PPP, highlighting the symptoms and potential impacts. This proactive approach aims to demystify these conditions, encouraging affected mothers to seek help without stigma. Official guidelines, such as those from Womenshealth.gov, advocate for mothers to consult healthcare providers if symptoms persist for over two weeks or worsen within a year post-delivery.

The involvement of partners, family members, and close associates is crucial in recognizing and responding to postpartum mental health issues. They should be informed about the signs of PPD and PPP and encouraged to assist the affected mother in accessing necessary care.

Furthermore, the American College of Obstetricians and Gynecologists (ACOG) emphasizes the importance of postpartum check-ups within the first 12 weeks after childbirth. These visits should extend beyond physical assessments to include open discussions about mental well-being, offering a safe space for new mothers to express any concerns about PPD symptoms. Healthcare providers can then offer appropriate interventions, including medication and mental health services, tailored to the individual's needs.
'''