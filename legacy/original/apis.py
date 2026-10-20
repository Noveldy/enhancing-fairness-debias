import logging
from main import check_dataset_bias
from flask import Flask, request, Response, jsonify, send_file
import bias_mitigation.synthetic_data.value_functions as vf
from bias_mitigation.synthetic_data.synthetic_data_generator import AdvancedSyntheticDataGenerator, TimeSeriesSyntheticGenerator, SyntheticIO
from bias_mitigation.evaluation_metrics.evaluation_recommender import recommend_evaluation_metrics
from bias_mitigation.optimizers.optimizer_recommendation import optimizer_recommender
import inspect
import ast
import textwrap
import importlib.util
import io
from datetime import datetime
from time import time
import numpy as np
import random
from werkzeug.utils import secure_filename
import requests
import json
from flask_cors import CORS, cross_origin
import sys
import os
import pandas as pd
from pathlib import Path
# app config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
VALUE_FUNC_FILE = Path("bias_mitigation/synthetic_data/value_functions.py")
SCHEMA_FILE = Path("bias_mitigation/synthetic_data/data_schemas.json")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#this is the data bias API
@app.route('/data-bias-api', methods=['POST'])
@cross_origin()
def data_bias_api():
    if 'file' not in request.files:
        return Response('No file part', status=500, mimetype="text/plain")
    # get the file from the request
    file = request.files['file']
    if file.filename == '':
        return Response('No file was selected', status=500, mimetype="text/plain")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # check if the file is a .csv
        if filename.endswith('.csv'):
            # load the data from the file
            data = pd.read_csv(Path(f'uploads/{filename}'))
        elif filename.endswith('.json'):
            # load the data as a json file
            with open(Path(f'uploads/{filename}'), 'r') as f:
                data = json.load(f)
            # check if the data key is present in the json
            if 'data' in data:
                # if the data key is present, get the value of the data key
                data = data['data']
            # convert the string to json
            json_data = json.loads(data)
            # convert the json data to a pandas dataframe
            data = pd.DataFrame(json_data)
        # get the values from the form fields
        detect_outliers = eval(request.form['detect_outliers'])
        z_thresh = int(request.form['z_thresh'])
        target_variables = request.form['target_variables']
        target_variables_descriptions = request.form['target_variables_descriptions']
        features = request.form['features']
        features_descriptions = request.form['features_descriptions']
        # split the request form fields into lists
        target_variables = target_variables.split(',')
        target_variables_descriptions = target_variables_descriptions.split(',')
        features = features.split(',')
        features_descriptions = features_descriptions.split(',')
        # empty logs.txt file
        with open(Path(f'results/logs.txt'), 'w') as log_file:
            log_file.write('')
        try:
            # remove file extension
            filename = filename.split('.')[0]         
            # call check dataset bias function
            mechanism_response = check_dataset_bias(data, filename, detect_outliers, z_thresh, target_variables, target_variables_descriptions, features, features_descriptions)
            # get the debias_ops_log key from the response
            debias_ops_log = mechanism_response['debias_ops_log']
            # remove the key from the response
            del mechanism_response['debias_ops_log']
            # split the log per new line
            debias_ops_log = debias_ops_log.split('\n')
            # remove duplicates from the log
            debias_ops_log = list(dict.fromkeys(debias_ops_log))
            # convert the log to a string
            debias_ops_log = '\n'.join(debias_ops_log)
            # create a new key in the response
            mechanism_response['debias_ops_log'] = debias_ops_log
            return Response(json.dumps(mechanism_response), status=200, mimetype="application/json")
        except Exception as e:
            return Response(e, status=500, mimetype="text/plain")

# this is the API for retrieving all the value functions
@app.route('/get-value-functions-api', methods=['GET'])
@cross_origin()
def list_value_functions():
    functions_info = []

    for name, func in vf.VALUE_FUNC_FACTORY_REGISTRY.items():
        # Use provided _params if available, otherwise inspect
        if hasattr(func, "_params"):
            param_list = func._params
        else:
            # Auto-detect parameter names from the factory function's 'params' argument
            try:
                # Inspect the inner lambda function this factory returns
                factory_func = func({})
                source = inspect.getsource(factory_func)
                # This is a fallback: if structured docs aren't defined, just skip detailed param parsing
                param_list = []
            except Exception:
                param_list = []

        functions_info.append({
            "name": name,
            "doc": func.__doc__.strip() if func.__doc__ else "No description available.",
            "params": param_list
        })

    return jsonify(functions_info)

# this is the API for submitting a new value function
@app.route('/submit-value-function-api', methods=['POST'])
@cross_origin()
def submit_value_function():
    code = request.json.get("code")

    if not code:
        return jsonify({"success": False, "error": "No code provided"}), 400

    # Check for syntax errors
    try:
        ast.parse(code)
    except SyntaxError as e:
        return jsonify({"success": False, "error": f"Syntax error: {str(e)}"}), 400

    if "@register_value_func" not in code:
        return jsonify({"success": False, "error": "Missing @register_value_func decorator."}), 400

    # Safe execution context
    context = {
        "np": np,
        "random": random,
        "register_value_func": lambda name=None: lambda func: func,  # dummy decorator
    }

    try:
        # Execute code safely
        exec(code, context)

        # Find the function ending with _factory
        factory_items = [(k, v) for k, v in context.items() if callable(v) and k.endswith("_factory")]
        if not factory_items:
            return jsonify({"success": False, "error": "No valid factory function found (must end with '_factory')."}), 400

        func_name, factory = factory_items[0]

        # Check if the function alredy exists in the value_functions.py file
        with open(VALUE_FUNC_FILE, "r") as f:
            existing_code = f.read()
            if func_name in existing_code:
                return jsonify({"success": False, "error": f"Function '{func_name}' already exists in the file."}), 400

        # Docstring check
        if not factory.__doc__ or not factory.__doc__.strip():
            return jsonify({"success": False, "error": f"Factory '{func_name}' is missing a docstring."}), 400

        # _params check
        if not hasattr(factory, "_params"):
            return jsonify({"success": False, "error": f"Factory '{func_name}' must define a `_params` attribute."}), 400
        if not isinstance(factory._params, list):
            return jsonify({"success": False, "error": "`_params` must be a list."}), 400

        # Run test
        value_func = factory({})
        test_output = value_func(5)

    except Exception as e:
        return jsonify({"success": False, "error": f"Function test failed: {str(e)}"}), 400

    # Append to file if valid
    try:
        with open(Path("bias_mitigation/synthetic_data/value_functions.py"), "a") as f:
            f.write("\n\n" + textwrap.dedent(code.strip()))
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to write to file: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "message": f"Function '{func_name}' successfully validated and saved.",
        "test_output": test_output
    })

# this is the API for submitting a data schema
@app.route('/submit-data-schema-api', methods=['POST'])
@cross_origin()
def submit_data_schema():
    data = request.get_json()

    if not data or "name" not in data or "schema" not in data:
        return jsonify({"success": False, "error": "Request must contain 'name' and 'schema' fields."}), 400

    name = data["name"]
    schema = data["schema"]

    if not isinstance(name, str) or not isinstance(schema, dict):
        return jsonify({"success": False, "error": "'name' must be a string and 'schema' must be a dictionary."}), 400

    # Basic schema validation (extensible)
    for field, config in schema.items():
        if "type" not in config:
            return jsonify({"success": False, "error": f"Missing 'type' for field '{field}'."}), 400

        if config["type"] == "timeseries":
            required_keys = {"start", "periods", "freq", "value_func", "entity_col"}
            missing = required_keys - config.keys()
            if missing:
                return jsonify({"success": False, "error": f"Missing keys {missing} in timeseries field '{field}'."}), 400

            # Value func check
            vf = config.get("value_func", {})
            if "name" not in vf or "params" not in vf:
                return jsonify({"success": False, "error": f"'value_func' must contain 'name' and 'params' in field '{field}'."}), 400
            
    # check if the schema can create a dataframe
    try:
        try:
            generator = TimeSeriesSyntheticGenerator(schema)
            df = generator.generate(n_rows=1)
            if not isinstance(df, pd.DataFrame):
                return jsonify({"success": False, "error": f"Generated data is not a DataFrame for schema '{name}'."}), 400
        except:
            generator = AdvancedSyntheticDataGenerator(schema)
            df = generator.generate(n_rows=1)
            if not isinstance(df, pd.DataFrame):
                return jsonify({"success": False, "error": f"Generated data is not a DataFrame for schema '{name}'."}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Provided schema is not valid."}), 500

    # Ensure directory exists
    SCHEMA_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing schemas
    if SCHEMA_FILE.exists():
        with open(SCHEMA_FILE, "r") as f:
            try:
                all_schemas = json.load(f)
            except json.JSONDecodeError:
                all_schemas = {}
    else:
        all_schemas = {}

    # Check for name collision
    if name in all_schemas:
        return jsonify({"success": False, "error": f"Schema name '{name}' already exists."}), 400

    # Save the new schema
    all_schemas[name] = schema
    with open(SCHEMA_FILE, "w") as f:
        json.dump(all_schemas, f, indent=2)

    return jsonify({"success": True, "message": f"Schema '{name}' saved successfully."}), 200

@app.route('/get-data-schemas-api', methods=['GET'])
@cross_origin()
def get_data_schemas():
    # If a specific schema name is provided in query parameters
    schema_name = request.args.get('name')

    # Check if the schemas file exists
    if not SCHEMA_FILE.exists():
        return jsonify({"success": False, "error": "No data schemas available."}), 404

    # Load the existing schemas
    with open(SCHEMA_FILE, "r") as f:
        try:
            all_schemas = json.load(f)
        except json.JSONDecodeError:
            return jsonify({"success": False, "error": "Failed to load schemas."}), 500

    if schema_name:
        # Return a specific schema if requested
        schema = all_schemas.get(schema_name)
        if not schema:
            return jsonify({"success": False, "error": f"Schema '{schema_name}' not found."}), 404
        return jsonify({
            "success": True,
            "schema_name": schema_name,
            "schema": schema
        }), 200

    # Return all schemas if no name is specified
    return jsonify({
        "success": True,
        "schemas": list(all_schemas.keys())
    }), 200


# this is the API for generating synthetic data
@app.route('/generate-synthetic-data-api', methods=['POST'])
@cross_origin()
def generate_synthetic_data():
    data = request.get_json()

    # Validate input data
    if not data or "schema_name" not in data:
        return jsonify({"success": False, "error": "Request must contain 'schema_name'."}), 400

    schema_name = data["schema_name"]
    n_rows = data["n_rows"] if "n_rows" in data else 10  # Default to 10 rows if not specified

    # Load schemas
    if not SCHEMA_FILE.exists():
        return jsonify({"success": False, "error": "No data schemas available."}), 404

    with open(SCHEMA_FILE, "r") as f:
        try:
            all_schemas = json.load(f)
        except json.JSONDecodeError:
            return jsonify({"success": False, "error": "Failed to load schemas."}), 500

    # Check if the schema exists
    if schema_name not in all_schemas:
        return jsonify({"success": False, "error": f"Schema '{schema_name}' not found."}), 404

    # Get the selected schema
    schema = all_schemas[schema_name]
    # check if the timeseries_activity is in the schema
    if "timeseries_activity" in schema:
        # Initialize TimeSeriesSyntheticGenerator
        generator = TimeSeriesSyntheticGenerator(schema)
        generator_type = "TimeSeriesSyntheticGenerator"
    else:
        # Initialize AdvancedSyntheticDataGenerator
        generator = AdvancedSyntheticDataGenerator(schema)
        generator_type = "AdvancedSyntheticDataGenerator"

    # Generate synthetic data
    try:
        generated_data = generator.generate(n_rows=n_rows)
        # convert datetime columns to string format if they exist
        if isinstance(generated_data, pd.DataFrame):
            for col in generated_data.select_dtypes(include=['datetime64']).columns:
                generated_data[col] = generated_data[col].astype(str)
        # If the generated data is a DataFrame, convert it to a JSON-serializable format
        if isinstance(generated_data, pd.DataFrame):
            generated_data = generated_data.replace({np.nan: None})
            # generate timestamp for the HTML and dataset files
            timestamp = int(time())
            # get the html file path
            html_file_name = f'{timestamp}_{schema_name}_{generator_type}_plots.html'
            html_file_path = f'plots/{html_file_name}'
            dataset_file_path = f'Datasets/{timestamp}_{schema_name}_{generator_type}_dataset.json'
            # call the audit bias function to generate the bias audit report
            SyntheticIO.audit_bias(generated_data, 10, True, 'plots', html_file_name)
            generated_data = generated_data.to_dict(orient="records")
            # store the generated data in a .json file
            with open(dataset_file_path, 'w') as dataset_file:
                json.dump(generated_data, dataset_file, indent=2)
        # Add metadata: date and time of creation
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return jsonify({"success": False, "error": f"Error generating data: {str(e)}"}), 500
    


    return jsonify({
        "success": True,
        "schema_name": schema_name,
        "generator_type": generator_type,
        "generated_data_path": dataset_file_path,
        "metadata": {
            "created_at": created_at
        },
        "html_file_path": html_file_path
    }), 200

# this is the API for retrieving data stored in the server based on the file path
@app.route('/get-data-api', methods=['GET'])
@cross_origin()
def get_data():
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({"success": False, "error": "No file path provided."}), 400

    # Check if the file exists
    if not os.path.exists(file_path):
        return jsonify({"success": False, "error": f"File '{file_path}' not found."}), 404

    # Read the data from the file
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return jsonify({
            "success": True,
            "data": data
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": f"Error reading file: {str(e)}"}), 500

# this is the API for retrieving a plot from the plots folder, given its path
@app.route('/get-plot-api', methods=['GET'])
@cross_origin()
def get_plot():
    plot_path = request.args.get('path')
    if not plot_path:
        return jsonify({"success": False, "error": "No plot path provided."}), 400

    # Check if the file exists
    if not os.path.exists(plot_path):
        return jsonify({"success": False, "error": f"Plot '{plot_path}' not found."}), 404

    # send the file as a response with send_file
    return send_file(plot_path, mimetype='text/html', as_attachment=False)

# this is the API for recommending optimizers
@app.route('/recommend-optimizer-api', methods=['POST'])
@cross_origin()
def recommend_optimizer_api():
    data = request.get_json()
    if not data or "problem_statement" not in data:
        return jsonify({"success": False, "error": "No problem statement provided."}), 400

    problem_statement = data["problem_statement"]
    dataset_size = data.get("dataset_size", None)
    try:
        recommendations = optimizer_recommender(problem_statement, dataset_size)
        return recommendations
    except Exception as e:
        return jsonify({"success": False, "error": f"Error during recommendation: {str(e)}"}), 500

# this is the API for recommending evaluation metrics
@app.route('/recommend-evaluation-metrics-api', methods=['POST'])
@cross_origin()
def recommend_evaluation_metrics_api():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"success": False, "error": "No query provided."}), 400

    query = data["query"]
    try:
        json_output = recommend_evaluation_metrics(query)
        return Response(json_output, status=200, mimetype="application/json")
    except Exception as e:
        return jsonify({"success": False, "error": f"Error during recommendation: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)