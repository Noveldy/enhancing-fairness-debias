import pandas as pd
import time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, pipeline
from huggingface_hub import PyTorchModelHubMixin

# visualizations
import plotly.express as px
import kaleido
import plotly.graph_objects as go

class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)
    
# Setup configuration and model
# Model info available here: https://huggingface.co/nvidia/domain-classifier
config = AutoConfig.from_pretrained("nvidia/domain-classifier")
tokenizer = AutoTokenizer.from_pretrained("nvidia/domain-classifier")
model = CustomModel.from_pretrained("nvidia/domain-classifier")
model.eval()

def initial_domain_classification():
    # query used for scopus export:
    # ( TITLE-ABS-KEY ( "neural network" ) OR TITLE-ABS-KEY ( "machine learning" ) ) AND TITLE-ABS-KEY ( "optimizer" ) AND ( TITLE-ABS-KEY ( "SGD" ) OR TITLE-ABS-KEY ( "ASGD" ) OR TITLE-ABS-KEY ( "Nadam" ) OR TITLE-ABS-KEY ( "Adam" ) OR TITLE-ABS-KEY ( "RMSprop" ) OR TITLE-ABS-KEY ( "Rprop" ) OR TITLE-ABS-KEY ( "Adadelta" ) OR TITLE-ABS-KEY ( "Adagrad" ) OR TITLE-ABS-KEY ( "AdamW" ) OR TITLE-ABS-KEY ( "Adamax" ) OR TITLE-ABS-KEY ( "LBFGS" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) )
    # load the scopus_optimizers.csv file
    scopus_optimizers_df = pd.read_csv(Path('bias_mitigation/scopus/scopus_export_Nov 5-2025_50fb902e-1fdb-40cb-a9f1-e55b8ad02490.csv'))
    # keep the Title and Abstract columns
    scopus_optimizers_df = scopus_optimizers_df[['Title', 'Abstract', 'Author Keywords',"Index Keywords"]]
    # convert all columns to lowercase
    scopus_optimizers_df = scopus_optimizers_df.applymap(lambda s: s.lower() if type(s) == str else s)
    # add an empty column called 'Domain'
    scopus_optimizers_df['Domain'] = ''
    # add an empty column called 'Optimizers'
    scopus_optimizers_df['Optimizers'] = ''
    # create a list of possible domains to which the papers can belong
    possible_domains = ['STEM (Science, Technology, Engineering and Mathematics)', 'Life and Health Sciences', 'Social Sciences', 'Humanities and Arts']
    # optimizers list
    optimizers = ['Adam', 'SGD', 'RMSprop', 'Rprop', 'Adagrad', 'Adadelta', 'AdamW', 'Nadam', 'FTRL', 'L-BFGS', 'LBFGS', 'ASGD']
    # convert optimizers to lowercase
    optimizers = [optimizer.lower() for optimizer in optimizers]
    # iterate over the scopus_optimizers_df and classify each abstract
    for index, row in scopus_optimizers_df.iterrows():
        for optimizer in optimizers:
            if optimizer in str(row['Title']) or optimizer in str(row['Abstract']) or optimizer in str(row['Author Keywords']) or optimizer in str(row['Index Keywords']):
                if scopus_optimizers_df.at[index, 'Optimizers'] == '':
                    scopus_optimizers_df.at[index, 'Optimizers'] = optimizer
                else:
                    scopus_optimizers_df.at[index, 'Optimizers'] += ';' + optimizer
    # print(scopus_optimizers_df['Optimizers'])
    # print(scopus_optimizers_df['Optimizers'].unique())
    # count total number of rows that contain ; in the Optimizers column
    count_multiple_optimizers = scopus_optimizers_df['Optimizers'].str.contains(';').sum()
    # count total number of unique values in the Optimizers column
    unique_optimizers = scopus_optimizers_df['Optimizers'].nunique()
    print(f'Number of unique optimizer combinations: {unique_optimizers}')
    # print frequency of each unique value in the Optimizers column
    print(scopus_optimizers_df['Optimizers'].value_counts())
    print(f'Number of papers with multiple optimizers: {count_multiple_optimizers}')
    print(f'Total number of papers: {len(scopus_optimizers_df)}')
    # for loop to classify each abstract
    for index, row in scopus_optimizers_df.iterrows():
        start_time = time.time()
        if row['Index Keywords'] and type(row['Index Keywords']) == str:
            text_to_classify = row['Index Keywords'].replace(';', ' ')
        elif row['Author Keywords'] and type(row['Author Keywords']) == str:
            text_to_classify = row['Author Keywords'].replace(';', ' ')
        elif row['Abstract'] and type(row['Abstract']) == str:
            text_to_classify = row['Abstract'].replace(';', ' ')
        else:
            text_to_classify = row['Title']
        # Prepare and process inputs
        text_samples = [text_to_classify]
        inputs = tokenizer(text_samples, return_tensors="pt", padding="longest", truncation=True)
        outputs = model(inputs["input_ids"], inputs["attention_mask"])

        # Predict and display results
        predicted_classes = torch.argmax(outputs, dim=1)
        predicted_domains = [config.id2label[class_idx.item()] for class_idx in predicted_classes.cpu().numpy()]
        predicted_domain = predicted_domains[0]
        scopus_optimizers_df.at[index, 'Domain'] = predicted_domain
        end_time = time.time()
        print(f'Classified paper {index + 1}/{len(scopus_optimizers_df)} as {predicted_domain} in {end_time - start_time:.2f} seconds.') # 0.25 seconds per paper on average (3542 papers total)

    # save the dataframe to a new csv file
    scopus_optimizers_df.to_csv(Path('bias_mitigation/scopus/scopus_optimizers_domain_classified.csv'), index=False)

def remove_papers_without_abstract():
    # load the scopus_optimizers_domain_classified.csv file
    scopus_optimizers_df = pd.read_csv(Path('bias_mitigation/scopus/scopus_optimizers_domain_classified.csv'))
    # drop rows where Abstract is equal to ''
    scopus_optimizers_df = scopus_optimizers_df[scopus_optimizers_df['Abstract'] != '[no abstract available]']
    print(f'Number of papers is {len(scopus_optimizers_df)}')
    # save the dataframe to a new csv file
    scopus_optimizers_df.to_csv(Path('bias_mitigation/scopus/scopus_optimizers_domain_classified_no_empty_abstracts.csv'), index=False)

def reduce_classification_categories():
    # load the scopus_optimizers_domain_classified.csv file
    scopus_optimizers_df = pd.read_csv(Path('bias_mitigation/scopus/scopus_optimizers_domain_classified.csv'))
    print(scopus_optimizers_df['Domain'].unique())
    category_map = {
        "Computers_and_Electronics": "STEM",
        "Science": "STEM",
        "Jobs_and_Education": "STEM",
        "Business_and_Industrial": "STEM",
        "Internet_and_Telecom": "STEM",

        "Health": "Life & Health Sciences",
        "Beauty_and_Fitness": "Life & Health Sciences",
        "Food_and_Drink": "Life & Health Sciences",
        "Pets_and_Animals": "Life & Health Sciences",

        "People_and_Society": "Social Sciences",
        "Finance": "Social Sciences",
        "News": "Social Sciences",

        "Arts_and_Entertainment": "Humanities & Arts",
        "Games": "Humanities & Arts",

        "Travel_and_Transportation": "Multidisciplinary",
        "Home_and_Garden": "Multidisciplinary",
        "Real_Estate": "Multidisciplinary",
        "Sports": "Multidisciplinary",
        "Online_Communities": "Multidisciplinary",
        "Sensitive_Subjects": "Multidisciplinary",
        "Autos_and_Vehicles": "Multidisciplinary"
    }
    scopus_optimizers_df['Reduced Domain'] = scopus_optimizers_df['Domain'].map(category_map)
    # drop Domain column
    scopus_optimizers_df = scopus_optimizers_df.drop(columns=['Domain'])
    # rename Reduced Domain column to Domain
    scopus_optimizers_df = scopus_optimizers_df.rename(columns={'Reduced Domain': 'Domain'})
    # drop rows where Abstract is equal to ''
    scopus_optimizers_df = scopus_optimizers_df[scopus_optimizers_df['Abstract'] != '[no abstract available]']
    # print(f'Number of papers is {len(scopus_optimizers_df)}') Number of papers is now 3535
    # save the dataframe to a new csv file
    scopus_optimizers_df.to_csv(Path('bias_mitigation/scopus/scopus_optimizers_final_domain_classification.csv'), index=False)

# reduce_classification_categories()

def identify_selected_optimizer():
    # Load zero-shot classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    question = "Which optimizer was chosen?"
    
    # load the scopus_optimizers_final_domain_classification.csv file
    scopus_optimizers_df = pd.read_csv(Path('bias_mitigation/scopus/scopus_optimizers_final_domain_classification.csv'))
    # drop rows where Optimizers is NaN
    scopus_optimizers_df = scopus_optimizers_df[scopus_optimizers_df['Optimizers'].notna()]
    # for loop to iterate over the scopus_optimizers_df
    for index, row in scopus_optimizers_df.iterrows():
        if ';' in row['Optimizers']:
            start_time = time.time()
            # print(f'Processing paper {index + 1}/{len(scopus_optimizers_df)}')
            choices = row['Optimizers'].split(';')
            context = row['Abstract']
            # Turn choices into natural-language hypotheses
            hypotheses = [f"Answer to the question '{question}' is {c}." for c in choices]
            # Get probabilities
            result = classifier(context, candidate_labels=hypotheses)
            # Pick best choice
            best_idx = result['scores'].index(max(result['scores']))
            # print("Predicted answer:", choices[best_idx])
            scopus_optimizers_df.at[index, 'Optimizers'] = choices[best_idx]
            end_time = time.time()
            print(f'Selected optimizer for paper {index + 1}/{len(scopus_optimizers_df)} in {end_time - start_time:.2f} seconds.') # average 6 seconds per paper
    # save the dataframe to a new csv file
    scopus_optimizers_df.to_csv(Path('bias_mitigation/scopus/scopus_optimizers_final_selected.csv'), index=False)

def visualize_subcategories_optimizers_combinations():
    # ------------------------ Subcategories Utilized Optimizers Visualization ------------------------ #
    # load the scopus_optimizers_domain_classified_no_empty_abstracts.csv file
    scopus_optimizers_final_domain_classification_df = pd.read_csv(Path('bias_mitigation/scopus/scopus_optimizers_domain_classified_no_empty_abstracts.csv'))
    # drop rows with NaN values
    scopus_optimizers_final_domain_classification_df.dropna(inplace=True)
    # get unique values in the Domain column
    unique_domains = scopus_optimizers_final_domain_classification_df['Domain'].unique()
    # get unique optimizers combinations
    unique_optimizers_combinations = scopus_optimizers_final_domain_classification_df['Optimizers'].unique()
    # merge unique optimizers combinations into a single string
    merged_optimizers_combinations = ';'.join(unique_optimizers_combinations)
    # print(f'Unique optimizer combinations: {merged_optimizers_combinations}')
    # find unique words in the merged_optimizers_combinations string
    optimizers = set(merged_optimizers_combinations.split(';'))
    # empty dataframe to store domain name, optimizer name, and count
    optimizer_visualization_df = pd.DataFrame(columns=['Domain', 'Optimizer', 'Count'])
    for domain in unique_domains:
        print(f'Visualizing domain: {domain}')
        # create an empty dictionary to hold optimizer counts
        optimizer_counts = {optimizer: 0 for optimizer in optimizers}
        # filter dataframe for rows with this domain
        filtered_df = scopus_optimizers_final_domain_classification_df[scopus_optimizers_final_domain_classification_df['Domain'] == domain]
        for index, row in filtered_df.iterrows():
            for optimizer in optimizers:
                if optimizer in str(row['Optimizers']):
                    optimizer_counts[optimizer] += 1
        # rename keys in optimizer_counts based on a mapping
        key_mapping = {'sgd': 'SGD', 'asgd': 'ASGD', 'nadam': 'Nadam', 'adam': 'Adam', 'rmsprop': 'RMSprop', 'rprop': 'Rprop', 'adadelta': 'Adadelta', 'adagrad': 'Adagrad', 'adamw': 'AdamW', 'adamax': 'Adamax', 'lbfgs': 'LBFGS', 'ftrl': 'FTRL'}
        optimizer_counts = {key_mapping.get(k, k): v for k, v in optimizer_counts.items()}
        # order optimizer_counts by keys alphabetically
        optimizer_counts = dict(sorted(optimizer_counts.items()))
        # convert optimizer_counts to a pandas series
        optimizer_counts = pd.Series(optimizer_counts)
        # drop optimizers with zero count
        non_zero_optimizer_counts = optimizer_counts[optimizer_counts > 0]
        # create bar chart
        fig = px.bar(non_zero_optimizer_counts, x=non_zero_optimizer_counts.index, y=non_zero_optimizer_counts.values, title=f'Distribution of utilized optimizers in research papers related to the <br>"{domain}" subdomain')
        fig.update_layout(xaxis_title='Optimizers', yaxis_title='Number of Papers', title_x=0.5)
        # set colors
        color_map = {
            'ASGD': 'RoyalBlue',
            'Adadelta': 'MediumSlateBlue',
            'Adagrad': 'CadetBlue',
            'Adamax': 'CornflowerBlue',
            'AdamW': 'DarkBlue',
            'Adam': 'DodgerBlue',
            'LBFGS': 'LightSkyBlue',
            'Nadam': 'Cyan',
            'RMSprop': 'DarkGreen',
            'Rprop': 'DarkCyan',
            'SGD': 'LightSeaGreen'
        }
        # define bar color based on optimizer name
        bar_colors = [color_map.get(opt, 'gray') for opt in non_zero_optimizer_counts.keys()]
        # set bar colors
        fig.update_traces(marker=dict(color=bar_colors))
        # store the figure as a png file in optimizers_subcategories_visualizations folder
        fig.write_image(f'bias_mitigation/scopus/optimizers_subcategories_visualizations/optimizers_distribution_{domain.replace(" ", "_")}.png')
        # concat the optimizer_counts to the optimizer_visualization_df
        temp_df = pd.DataFrame({'Domain': [domain]*len(optimizer_counts), 'Optimizer': optimizer_counts.index, 'Count': optimizer_counts.values})
        optimizer_visualization_df = pd.concat([optimizer_visualization_df, temp_df], ignore_index=True)
    # store the optimizer_counts as a csv file
    optimizer_visualization_df.to_csv(f'bias_mitigation/scopus/optimizers_subcategories_visualizations/raw_data/visualization_data.csv', index=False)

def visualize_main_categories_optimizers_combinations():
    # ------------------------ Main Categories Utilized Optimizers Visualization ------------------------ #
    # load the scopus_optimizers_final_domain_classification.csv file
    scopus_optimizers_final_domain_classification_df = pd.read_csv(Path('bias_mitigation/scopus/scopus_optimizers_final_domain_classification.csv'))
    # drop rows with NaN values
    scopus_optimizers_final_domain_classification_df.dropna(inplace=True)
    # get unique optimizers combinations
    unique_optimizers_combinations = scopus_optimizers_final_domain_classification_df['Optimizers'].unique()
    # merge unique optimizers combinations into a single string
    merged_optimizers_combinations = ';'.join(unique_optimizers_combinations)
    # find unique words in the merged_optimizers_combinations string
    optimizers = set(merged_optimizers_combinations.split(';'))
    # empty dataframe to store domain name, optimizer name, and count
    optimizer_visualization_main_df = pd.DataFrame(columns=['Domain', 'Optimizer', 'Count'])
    # iterate over scopus_optimizers_final_domain_classification_df
    for index, row in scopus_optimizers_final_domain_classification_df.iterrows():
        domain = row['Domain']
        optimizers = row['Optimizers'].split(';')
        for optimizer in optimizers:
            temp_df = pd.DataFrame({'Domain': [domain], 'Optimizer': [optimizer], 'Count': [1]})
            optimizer_visualization_main_df = pd.concat([optimizer_visualization_main_df, temp_df], ignore_index=True)
    # group by Domain and Optimizer and sum the Count
    optimizer_visualization_main_df = optimizer_visualization_main_df.groupby(['Domain', 'Optimizer'], as_index=False).sum()
    # map optimizer names to proper case
    optimizer_visualization_main_df['Optimizer'] = optimizer_visualization_main_df['Optimizer'].map({'sgd': 'SGD', 
                                                                                                     'asgd': 'ASGD', 'nadam': 'Nadam', 'adam': 'Adam', 
                                                                                                     'rmsprop': 'RMSprop', 'rprop': 'Rprop', 'adadelta': 'Adadelta', 
                                                                                                     'adagrad': 'Adagrad', 'adamw': 'AdamW', 'adamax': 'Adamax', 
                                                                                                     'lbfgs': 'LBFGS', 'ftrl': 'FTRL'})
    print(optimizer_visualization_main_df)
    domains = optimizer_visualization_main_df['Domain'].unique()
    for domain in domains:
        filtered_df = optimizer_visualization_main_df[optimizer_visualization_main_df['Domain'] == domain]
        # create bar chart
        fig = px.bar(filtered_df, x='Optimizer', y='Count',
                     labels={'Count': 'Number of Papers', 'Optimizer': 'Optimizers'},
                    title=f'Distribution of utilized optimizers in research papers related to the <br>"{domain}" domain')
        fig.update_layout(title_x=0.5)
        # set colors
        color_map = {
            'ASGD': 'RoyalBlue',
            'Adadelta': 'MediumSlateBlue',
            'Adagrad': 'CadetBlue',
            'Adamax': 'CornflowerBlue',
            'AdamW': 'DarkBlue',
            'Adam': 'DodgerBlue',
            'LBFGS': 'LightSkyBlue',
            'Nadam': 'Cyan',
            'RMSprop': 'DarkGreen',
            'Rprop': 'DarkCyan',
            'SGD': 'LightSeaGreen'
        }
        # define bar color based on optimizer name
        bar_colors = [color_map.get(opt, 'gray') for opt in filtered_df['Optimizer']]
        # set bar colors
        fig.update_traces(marker=dict(color=bar_colors))
        # store the figure as a png file in optimizers_main_categories_visualizations folder
        fig.write_image(f'bias_mitigation/scopus/optimizers_main_categories_visualizations/optimizers_distribution_{domain}.png')
        # store the optimizer_counts as a csv file
    optimizer_visualization_main_df.to_csv('bias_mitigation/scopus/optimizers_main_categories_visualizations/raw_data/optimizer_distribution.csv', index=False)

def visualize_selected_optimizers_per_main_category():
    # ------------------------ Selected Optimizers Visualization ------------------------ #
    # load the scopus_optimizers_final_selected.csv file
    scopus_optimizers_final_selected_df = pd.read_csv(Path('bias_mitigation/scopus/scopus_optimizers_final_selected.csv'))
    # drop rows with NaN values
    scopus_optimizers_final_selected_df.dropna(inplace=True)
    # get unique values in the Domain column
    unique_domains = scopus_optimizers_final_selected_df['Domain'].unique()
    # get unique optimizers combinations
    optimizers = scopus_optimizers_final_selected_df['Optimizers'].unique()
    # empty dataframe to store domain name, optimizer name, and count
    optimizer_visualization_selected_df = pd.DataFrame(columns=['Domain', 'Optimizer', 'Count'])
    for domain in unique_domains:
        print(f'Visualizing domain (selected optimizers): {domain}')
        # create an empty dictionary to hold optimizer counts
        optimizer_counts = {optimizer: 0 for optimizer in optimizers}
        # filter dataframe for rows with this domain
        filtered_df = scopus_optimizers_final_selected_df[scopus_optimizers_final_selected_df['Domain'] == domain]
        for index, row in filtered_df.iterrows():
            optimizer = row['Optimizers']
            if optimizer in optimizer_counts:
                optimizer_counts[optimizer] += 1
        # rename keys in optimizer_counts based on a mapping
        key_mapping = {'sgd': 'SGD', 'asgd': 'ASGD', 'nadam': 'Nadam', 'adam': 'Adam', 'rmsprop': 'RMSprop', 'rprop': 'Rprop', 'adadelta': 'Adadelta', 'adagrad': 'Adagrad', 'adamw': 'AdamW', 'adamax': 'Adamax', 'lbfgs': 'LBFGS', 'ftrl': 'FTRL'}
        optimizer_counts = {key_mapping.get(k, k): v for k, v in optimizer_counts.items()}
        # order optimizer_counts by keys alphabetically
        optimizer_counts = dict(sorted(optimizer_counts.items()))
        # convert optimizer_counts to a pandas series
        optimizer_counts = pd.Series(optimizer_counts)
        # drop optimizers with zero count
        non_zero_optimizer_counts = optimizer_counts[optimizer_counts > 0]
        # create pie chart
        fig = px.pie(non_zero_optimizer_counts, names=non_zero_optimizer_counts.index, values=non_zero_optimizer_counts.values, title=f'Percentage of selected optimizers in research papers related to the <br>"{domain}" domain', color = non_zero_optimizer_counts.index, color_discrete_map={
            'ASGD': 'RoyalBlue',
            'Adadelta': 'MediumSlateBlue',
            'Adagrad': 'CadetBlue',
            'Adamax': 'CornflowerBlue',
            'AdamW': 'DarkBlue',
            'Adam': 'DodgerBlue',
            'LBFGS': 'LightSkyBlue',
            'Nadam': 'Cyan',
            'RMSprop': 'DarkGreen',
            'Rprop': 'DarkCyan',
            'SGD': 'LightSeaGreen'})
        fig.update_layout(title_x=0.5)
        # store the figure as a png file in optimizers_main_categories_visualizations folder
        fig.write_image(f'bias_mitigation/scopus/selected_optimizers_main_categories_visualizations/{domain}_optimizer_distribution.png')
        # concat the optimizer_counts to the optimizer_visualization_selected_df
        temp_df = pd.DataFrame({'Domain': [domain]*len(optimizer_counts), 'Optimizer': optimizer_counts.index, 'Count': optimizer_counts.values})
        optimizer_visualization_selected_df = pd.concat([optimizer_visualization_selected_df, temp_df], ignore_index=True)
    # store the optimizer_counts as a csv file
    optimizer_visualization_selected_df.to_csv('bias_mitigation/scopus/selected_optimizers_main_categories_visualizations/raw_data/visualization_data.csv', index=False)

# identify_selected_optimizer()
# remove_papers_without_abstract()
# visualize_subcategories_optimizers_combinations()
# visualize_main_categories_optimizers_combinations()
visualize_selected_optimizers_per_main_category()

