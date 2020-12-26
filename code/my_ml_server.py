import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from flask import Flask, render_template, url_for, request, redirect
from flask import send_from_directory
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm


from wtforms.validators import DataRequired, Optional
from wtforms import StringField, SubmitField, FileField, TextField, \
    RadioField, IntegerField, FloatField, TextAreaField

from ml_part import RandomForestMSE, GradientBoostingMSE

app = Flask(__name__, template_folder='html', static_folder='static')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
PEOPLE_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
data_path = './../data'
Bootstrap(app)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

X = {}
M = {}
M_info = {}
super_style = "\
    background: \
        -webkit-gradient(linear, 0 0, 0  100%, from(#800000), \
                         to(#ff4500), color-stop(0.5, #ff0000)); \
        -webkit-gradient(linear, 0 0, 0  100%, from(#D0ECF4), \
                         to(#D0ECF4), color-stop(0.5, #5BC9E1)); \
    filter: \
        progid: \
            DXImageTransform.Microsoft.gradient(startColorstr='#00BBD6', \
                                                endColorstr='#EBFFFF'); \
    padding: 6px 14px; \
    color: #000; \
    -webkit-border-radius: 10px; \
    border-radius: 10px; \
    border: 2px solid #666; \
    height: 150px; \
    width: 1000px; \
    font-size: 50pt;"
super_style2 = super_style.replace("-webkit-gradient(linear, 0 0, 0  100%, from(#800000), \
                         to(#ff4500), color-stop(0.5, #ff0000));", 
                            "-webkit-gradient(linear, 0 0, 0  100%, from(#D0ECF4), \
                         to(#D0ECF4), color-stop(0.5, #5BC9E1)); ").replace("50pt", "30pt").replace("height: 150px; \
    width: 1000px;", "height: 100px; width: 750px;")
super_style3 = super_style.replace("-webkit-gradient(linear, 0 0, 0  100%, from(#800000), \
                         to(#ff4500), color-stop(0.5, #ff0000));", 
                            "-webkit-gradient(linear, 0 0, 0  100%, from(#90EE90), \
                         to(#90EE90), color-stop(0.5, #F5F5F5)); ").replace("50pt", "20pt").replace("height: 150px; \
    width: 1000px;", "height: 100px; width: 400px;")
super_style4 = super_style.replace("-webkit-gradient(linear, 0 0, 0  100%, from(#800000), \
                         to(#ff4500), color-stop(0.5, #ff0000));", 
                            "-webkit-gradient(linear, 0 0, 0  100%, from(#006400), \
                         to(#006400), color-stop(0.5, #ADFF2F)); ").replace("50pt", "20pt").replace("height: 150px; \
    width: 1000px;", "height: 70px; width: 400px;")

 
class Gen(FlaskForm):
    title = TextField("", render_kw={'style': super_style, 'readonly': True})
    title.data = 'Сервер для обучения моделей'
    submit1 = SubmitField('Добавить новый датасет', render_kw={'style': super_style2, 'readonly':True})
    submit2 = SubmitField('Обучить модель', render_kw={'style': super_style2})
    submit3 = SubmitField('Посмотреть информацию о моделях', render_kw={'style': super_style2})
    submit5 = SubmitField('Получить предсказания', render_kw={'style': super_style2})
    
class Gen_models(FlaskForm):
    submit3 = SubmitField('Посмотреть информацию о моделях', render_kw={'style': super_style2})
    
class to_Gen(FlaskForm):
    submit = SubmitField('Вернуться на главную страницу', render_kw={'style': super_style2})
    
class Add(FlaskForm):
    title = TextField("", render_kw={'style': super_style, 'readonly': True})
    X_path_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    X_path = FileField('', validators=[DataRequired()], render_kw={'style': super_style3})
    y_path_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    y_path = FileField('', render_kw={'style': super_style3})
    name_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    name = StringField('', validators=[DataRequired()], render_kw={'style': super_style3})
    submit = SubmitField('Добавить датасет', render_kw={'style': super_style2})
    
class Model_Info(FlaskForm):
    title = TextField("", render_kw={'style': super_style, 'readonly': True})
    model_name_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    model_name = StringField('', validators=[DataRequired()], render_kw={'style': super_style3})
    data_name_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    data_name = StringField('', validators=[DataRequired()], render_kw={'style': super_style3})
    test_name_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    test_name = StringField('', validators=[DataRequired()], render_kw={'style': super_style3})
    model_type_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    model_type = StringField('', validators=[DataRequired()], render_kw={'style': super_style3})
    n_estimators_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    n_estimators = IntegerField('', validators=[DataRequired()], render_kw={'style': super_style3})
    learning_rate_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    learning_rate = FloatField('', validators=[DataRequired()], render_kw={'style': super_style3})
    #depth_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    #depth = IntegerField('', validators=[DataRequired()], render_kw={'style': super_style3})
    #feat_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    #feat = IntegerField('', validators=[DataRequired()], render_kw={'style': super_style3})
    
class For_fit(FlaskForm):
    title = TextField("", render_kw={'style': super_style, 'readonly': True})
    model_name_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    model_name = StringField('', validators=[DataRequired()], render_kw={'style': super_style3})
    train_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    train = RadioField('', choices=X.keys(), validators=[DataRequired()])
    test_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    test = RadioField('', choices=X.keys(), validators=[Optional()])
    model_type_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    model_type = RadioField('', choices=['Random_Forest', 'Gradient_Boosting'], validators=[DataRequired()])
    n_estimators_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    n_estimators = IntegerField('', validators=[DataRequired()], render_kw={'style': super_style3})
    lr_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    lr = FloatField('', validators=[DataRequired()], render_kw={'style': super_style3})
    #depth_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    #depth = IntegerField('', validators=[DataRequired()], render_kw={'style': super_style3})
    #feat_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    #feat = TextField('', validators=[DataRequired()], render_kw={'style': super_style3})
    submit = SubmitField('Обучить модель', render_kw={'style': super_style2})
    
class For_predict(FlaskForm):
    title = TextField("", render_kw={'style': super_style, 'readonly': True})
    model_name_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    model_name = RadioField('', choices=M.keys(), validators=[DataRequired()])
    test_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    test = RadioField('', choices=X.keys(), validators=[DataRequired()])
    res_title = TextField("", render_kw={'style': super_style4, 'readonly': True})
    res = StringField('', validators=[DataRequired()], render_kw={'style': super_style3})
    submit = SubmitField('Загрузить предсказания', render_kw={'style': super_style2})
    
class to_Picture(FlaskForm):
    submit = SubmitField('Показать статистику обучения', render_kw={'style': super_style2})

@app.route('/', methods=['GET', 'POST'])
def general():
    try:
        start_form = Gen()
        start_form.title.data = 'Сервер для обучения моделей'
        
        if start_form.validate_on_submit():
            if start_form.submit1.data:
                return redirect(url_for('add_data'))
            elif start_form.submit2.data:
                return redirect(url_for('make_fit'))
            elif start_form.submit3.data:
                return redirect(url_for('models_info'))
            elif start_form.submit5.data:
                return redirect(url_for('make_predict'))
        forms = [start_form]
        return render_template('forms.html', forms=forms)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    try:
        add_form = Add()
        add_form.title.data = 'Добавление нового датасета'
        add_form.X_path_title.data = 'Файл с объектами'
        add_form.y_path_title.data = 'Файл с целевым значением'
        add_form.name_title.data = 'Имя в системе'
        ret = to_Gen()
        
        if add_form.validate_on_submit():
            X_path = add_form.X_path.data
            y_path = add_form.y_path.data
            file_name = add_form.name.data
            global X       
            X[file_name] = {}
            pandas_X = pd.read_csv(X_path)
            X[file_name]['X'] = np.array(pandas_X)
            try:
                y_path = add_form.y_path.data
                pandas_y = pd.read_csv(y_path)
                X[file_name]['y'] = np.array(pandas_y)
            except Exception:
                return redirect(url_for('add_data'))
            
            return redirect(url_for('add_data'))
        
        elif ret.validate_on_submit():
            return redirect(url_for('general'))
        
        forms = [add_form, ret]
        return render_template('forms.html', forms=forms)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/models_info')
def models_info():
    return render_template('models_info.html', M=M)

@app.route('/show_model_info/picture/<name>')
def models_info2(name):
    return render_template('models_info.html', M=M)

@app.route('/show_model_info/<name>', methods=['GET', 'POST'])
def show_model_info(name):
    try:
        model = Model_Info()
        back = Gen_models()
        pic = to_Picture()
        model.title.data = 'Информация о модели'
        model.model_name_title.data = 'Имя модели'
        model.data_name_title.data = 'Имя датасета для обучения'
        model.test_name_title.data = 'Имя датасета для валидации'
        model.model_type_title.data = 'Тип модели'
        model.n_estimators_title.data = 'Число деревьев'
        model.learning_rate_title.data = 'Темп обучения'
        
        model.model_name.data = name
        model.model_type.data = M_info[name]['type']
        model.data_name.data = M_info[name]['train']
        if M_info[name]['val']:
            model.test_name.data = M_info[name]['test']
        else:
            model.test_name.data = 'Не задано'
            
        model.n_estimators.data = M[name].n
        
        if M_info[name]['type'] == 'Gradient_Boosting':
            model.learning_rate.data = M[name].l
        else:
            model.learning_rate.data = 'Не нужен'
        if pic.validate_on_submit():
            if pic.submit.data:
                return render_template('picture.html', \
                                       path=M_info[name]['pic'], name=name)
        
        if back.validate_on_submit():
            if back.submit3.data:
                return redirect(url_for('models_info'))
        
        if M_info[name]['val']:
            forms = [model, pic, back]
        else:
            forms = [model, back]
        return render_template('forms.html', forms=forms)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    
@app.route('/make_fit', methods=['GET', 'POST'])
def make_fit():
    try:
        fit_form = For_fit()
        fit_form.title.data = 'Обучение новой модели'
        fit_form.model_name_title.data = 'Имя модели'
        fit_form.train_title.data = 'Имя датасета для обучения'
        fit_form.test_title.data = 'Имя датасета для валидации'
        fit_form.model_type_title.data = 'Тип модели'
        fit_form.n_estimators_title.data = 'Число деревьев'
        fit_form.lr_title.data = 'Темп обучения'
        ret = to_Gen()
        
        if fit_form.validate_on_submit():
            if fit_form.submit.data:
                train = fit_form.train.data
                test = fit_form.test.data
                mtype = fit_form.model_type.data
                n_estimators = fit_form.n_estimators.data
                lr = fit_form.lr.data
                model_name = fit_form.model_name.data
    
                if mtype == 'Random_Forest':
                    model = RandomForestMSE(n_estimators=n_estimators)
                else:
                    model = GradientBoostingMSE(n_estimators=n_estimators, 
                                        learning_rate=lr)
            
                M_info[model_name] = {}
                if test is None:
                    model.fit(X[train]['X'], X[train]['y'])
                    M_info[model_name]['val'] = False
                    M[model_name] = model
                else:
                    model.fit(X[train]['X'], X[train]['y'], \
                              X[test]['X'], X[test]['y'])
                    M_info[model_name]['val'] = True
                    M_info[model_name]['RMSE'] = model.RMSE
                    M_info[model_name]['test'] = test
                    M[model_name] = model
                
                    fig, ax = plt.subplots(figsize=(15, 10))
                    
                    ax.set(facecolor = 'white')
                    ax.grid()
                    ax.tick_params(axis='x', labelsize=20)
                    ax.tick_params(axis='y', labelsize=20)
                    ax.set_xlabel('Номер итерации', fontsize=30)
                    ax.set_ylabel('Значение RMSE', fontsize=30)
                    ax.set_title('Динамика процесса обучения модели ' + \
                                 model_name, fontsize=30)
                    ax.plot(np.arange(M[model_name].n) + 1, \
                            M_info[model_name]['RMSE'], c='blue', linewidth=5);
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
                    path_of_script = os.path.abspath(__file__)
                    path_of_dir = path_of_script[:path_of_script.rfind('/') + 1]
                    plt.savefig(path_of_dir + 'static/' + model_name + '.jpg')
                    M_info[model_name]['pic'] = model_name + '.jpg'
            
                M_info[model_name]['type'] = mtype
                M_info[model_name]['train'] = train
                return redirect(url_for('make_fit'))
        elif ret.validate_on_submit():
            if ret.submit.data:
                return redirect(url_for('general'))
        forms = [fit_form, ret]
        return render_template('forms.html', forms=forms)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        
@app.route('/make_predict', methods=['GET', 'POST'])
def make_predict():
    try:
        predict_form = For_predict()
        ret = to_Gen()
        predict_form.title.data = 'Предсказание модели'
        predict_form.model_name_title.data = 'Имя модели'
        predict_form.test_title.data = 'Имя датасета'
        predict_form.res_title.data = 'Название файла загрузки'
        
        if predict_form.validate_on_submit():
            test = predict_form.test.data
            res = predict_form.res.data
            model_name = predict_form.model_name.data
            
            y_preds = pd.DataFrame(M[model_name].predict(X[test]['X']))
            path_of_script = os.path.abspath(__file__)
            path_of_dir = path_of_script[:path_of_script.rfind('/') + 1]
            y_preds.to_csv(path_of_dir + res)
            return redirect(url_for('make_predict'))
        elif ret.validate_on_submit():
            return redirect(url_for('general'))
        forms = [predict_form, ret]
        return render_template('forms.html', forms=forms)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))