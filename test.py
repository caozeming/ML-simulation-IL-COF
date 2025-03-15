from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 22
df = pd.read_excel("test_data.xlsx")

X = df.drop(columns=['Label']).values
y = df['Label'].values

rf_results = []
cat_results = []
dt_results = []
xgb_results = []
save_list = {}
save_list['Model_Name'] = []
save_list['mae'] = []
save_list['mse'] = []
save_list['rmse'] = []
save_list['r2'] = []

rf_model = joblib.load(filename=r'C:\Users\cao\Desktop\rf_model.pkl')
y_pred = rf_model.predict(X)
y_test = df['Label'].values
rf_results.append(('Random Forest', y_test, y_pred))
rf_result_pred = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})

cat_model = joblib.load(filename=r'C:\Users\cao\Desktop\catboost_model.pkl')
y_pred = cat_model.predict(X)
y_test = df['Label'].values
cat_results.append(('CatBoost', y_test, y_pred))
cat_result_pred = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})

dt_model = joblib.load(filename=r'C:\Users\cao\Desktop\dt_model.pkl')
dt_y_pred = dt_model.predict(X)
y_test = df['Label'].values
dt_results.append(('Decision Tree', y_test, dt_y_pred))
dt_result_pred = pd.DataFrame({'True Label': y_test, 'Predicted Label': dt_y_pred})

xgb_model = joblib.load(filename=r'C:\Users\cao\Desktop\xgb_model.pkl')
y_pred = xgb_model.predict(X)
y_test = df['Label'].values
xgb_results.append(('XGBoost', y_test, y_pred))
xgb_result_pred = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 22

def plot_results(model_name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    save_list['Model_Name'].append(model_name)
    save_list['mae'].append(mae)
    save_list['mse'].append(mse)
    save_list['rmse'].append(rmse)
    save_list['r2'].append(r2)
    print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    xy = np.vstack([y_test, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    y_test, y_pred, z = y_test[idx], y_pred[idx], z[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(y_test, y_pred, marker='o', c=z, edgecolors='none', s=45, cmap='Spectral_r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.01)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('Frequency', labelpad=20)
    cbar.ax.spines['top'].set_linewidth(4)
    cbar.ax.spines['bottom'].set_linewidth(4)
    cbar.ax.spines['left'].set_linewidth(4)
    cbar.ax.spines['right'].set_linewidth(4)
    ax.legend([f'{model_name} testing set'], loc='upper left')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    ax.set_xlabel('Simulated H$_2$ adsorption capacity (mmol/g)', labelpad=15)
    ax.set_ylabel('Predicted H$_2$ adsorption capacity (mmol/g)', labelpad=15)
    ax.set_title(f'{model_name} - True vs Predicted Density Plot', pad=20)
    ax.spines['left'].set_linewidth(2)  
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['right'].set_linewidth(2)  
    ax.spines['top'].set_linewidth(2)    
    ax.tick_params(axis='both', width=2) 
    plt.show()

    data = {'True_Values': y_test, 'Predicted_Values': y_pred}
    df = pd.DataFrame(data)
    df.to_excel(f'{model_name.lower().replace(" ", "_")}_test_pred.xlsx', index=False)

for model_name, y_test, y_pred in rf_results:
    plot_results(model_name, y_test, y_pred)

for model_name, y_test, y_pred in cat_results:
    plot_results(model_name, y_test, y_pred)

for model_name, y_test, y_pred in dt_results:
    plot_results(model_name, y_test, y_pred)

for model_name, y_test, y_pred in xgb_results:
    plot_results(model_name, y_test, y_pred)

xgb_result_pred['Predicted Label'] = xgb_result_pred['Predicted Label'].round(5)
dt_result_pred['Predicted Label'] = dt_result_pred['Predicted Label'].round(5)
rf_result_pred['Predicted Label'] = rf_result_pred['Predicted Label'].round(5)
cat_result_pred['Predicted Label'] = cat_result_pred['Predicted Label'].round(5)