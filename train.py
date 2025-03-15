from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import pickle
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

df = pd.read_excel("train_data.xlsx")
X = df.drop(columns=['Label']).values
y = df['Label'].values
df.sort_values(by="Label", inplace=True)


n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

rf_results = []
cat_results = []
dt_results = []
xgb_results = []
train_rf_results = []
train_cat_results = []
train_dt_results = []
train_xgb_results = []
save_list = {}
save_list['Model_Name'] = []
save_list['mae'] = []
save_list['mse'] = []
save_list['rmse'] = []
save_list['r2'] = []
rf_train_times = []
cat_train_times = []
dt_train_times = []
xgb_train_times = []

for train_index, test_index in kf.split(X):
    X_train_so, X_test_so = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = X_train_so
    X_test = X_test_so
    start_time = time.time()
    rf_model = RandomForestRegressor(n_estimators=90,random_state=42, verbose=0)
    rf_model.fit(X_train, y_train)
    end_time = time.time()
    rf_train_times.append(end_time - start_time)
    y_pred = rf_model.predict(X_test)
    rf_results.append(('Random Forest', y_test, y_pred))
    x_train_pred = rf_model.predict(X_train)
    train_rf_results.append(('Random Forest', y_train, x_train_pred))
    rf_result_train = pd.DataFrame({'y_train_true': y_train,'y_train_pred': x_train_pred})
    rf_result_pred = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})
    train_r2 = r2_score(y_train, x_train_pred)
    
    with open('Desktop/rf_model.pkl', 'wb') as file:
        pickle.dump(rf_model, file)

    start_time = time.time()    
    catboost_model = CatBoostRegressor(learning_rate=0.08, depth=5, iterations=1000, l2_leaf_reg=3, subsample=0.8, loss_function='RMSE',verbose=False,early_stopping_rounds=100)
    catboost_model.fit(X_train, y_train)
    end_time = time.time()
    cat_train_times.append(end_time - start_time)
    cat_y_pred = catboost_model.predict(X_test)
    cat_results.append(('CatBoost_model', y_test, cat_y_pred))
    x_train_pred = catboost_model.predict(X_train)
    train_cat_results.append(('CatBoost', y_train, x_train_pred))
    cat_result_train = pd.DataFrame({'y_train_true': y_train,'y_train_pred': x_train_pred})
    cat_result_pred = pd.DataFrame({'True Label': y_test, 'Predicted Label': cat_y_pred})

    with open('Desktop/catboost_model.pkl', 'wb') as file:
        pickle.dump(catboost_model, file)
    
    from sklearn.tree import DecisionTreeRegressor
    start_time = time.time()
    dt_model = DecisionTreeRegressor(max_depth=14, min_samples_split=2,criterion='absolute_error', max_features='log2',splitter='best', random_state=42, min_samples_leaf=2)
    dt_model.fit(X_train, y_train)
    end_time = time.time()
    dt_train_times.append(end_time - start_time)
    dt_y_pred = dt_model.predict(X_test)
    dt_results.append(('Decision Tree', y_test, dt_y_pred))
    x_train_pred = dt_model.predict(X_train)
    train_dt_results.append(('Decision Tree', y_train, x_train_pred))
    dt_result_train = pd.DataFrame({'y_train_true': y_train,'y_train_pred': x_train_pred})
    dt_result_pred = pd.DataFrame({'True Label': y_test, 'Predicted Label': dt_y_pred})

    with open('Desktop/dt_model.pkl', 'wb') as file:
        pickle.dump(dt_model, file)
        
    import xgboost as xgb
    start_time = time.time()
    xgb_model = xgb.XGBRegressor(verbosity=0,booster='gbtree', n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8)
    xgb_model.fit(X_train, y_train)
    end_time = time.time()
    xgb_train_times.append(end_time - start_time)
    xgb_y_pred = xgb_model.predict(X_test)
    xgb_results.append(('XGBoost', y_test, xgb_y_pred))
    x_train_pred = xgb_model.predict(X_train)
    train_xgb_results.append(('XGBoost', y_train, x_train_pred))
    xgb_result_train = pd.DataFrame({'y_train_true': y_train,'y_train_pred': x_train_pred})
    xgb_result_pred = pd.DataFrame({'True Label': y_test, 'Predicted Label': xgb_y_pred})

    with open('Desktop/xgb_model.pkl', 'wb') as file:
        pickle.dump(xgb_model, file)


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 22

def plot_train_results(model_name, y_train, x_train_pred):
    mae = mean_absolute_error(y_train, x_train_pred)
    mse = mean_squared_error(y_train, x_train_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, x_train_pred)
    save_list['Model_Name'].append(model_name)
    save_list['mae'].append(mae)
    save_list['mse'].append(mse)
    save_list['rmse'].append(rmse)
    save_list['r2'].append(r2)
    print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    xy = np.vstack([y_train, x_train_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    y_train, x_train_pred, z = y_train[idx], x_train_pred[idx], z[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(y_train, x_train_pred, marker='o', c=z, edgecolors='none', s=45, cmap='Spectral_r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.01)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('Frequency', labelpad=20)
    cbar.ax.spines['top'].set_linewidth(4)
    cbar.ax.spines['bottom'].set_linewidth(4)
    cbar.ax.spines['left'].set_linewidth(4)
    cbar.ax.spines['right'].set_linewidth(4)
    ax.legend([f'{model_name} training set'], loc='upper left')
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--')
    ax.set_xlabel('Simulated H$_2$ adsorption capacity (mmol/g)', labelpad=15)
    ax.set_ylabel('Predicted H$_2$ adsorption capacity (mmol/g)', labelpad=15)
    ax.set_title(f'{model_name} - True vs Predicted Density Plot', pad=20)
    ax.spines['left'].set_linewidth(2)  
    ax.spines['bottom'].set_linewidth(2)  
    ax.spines['right'].set_linewidth(2)  
    ax.spines['top'].set_linewidth(2)   
    ax.tick_params(axis='both', width=2)  
    plt.show()

    data = {'True_Values': y_train, 'Predicted_Values': x_train_pred}
    df = pd.DataFrame(data)
    df.to_excel(f'{model_name.lower().replace(" ", "_")}_train_pred.xlsx', index=False)

for model_name, y_train, x_train_pred in train_rf_results:
    plot_train_results(model_name, y_train, x_train_pred)

for model_name, y_train, x_train_pred in train_cat_results:
    plot_train_results(model_name, y_train, x_train_pred)

for model_name, y_train, x_train_pred in train_dt_results:
    plot_train_results(model_name, y_train, x_train_pred)

for model_name, y_train, x_train_pred in train_xgb_results:
    plot_train_results(model_name, y_train, x_train_pred)

xgb_result_pred['Predicted Label'] = xgb_result_pred['Predicted Label'].round(5)
dt_result_pred['Predicted Label'] = dt_result_pred['Predicted Label'].round(5)
rf_result_pred['Predicted Label'] = rf_result_pred['Predicted Label'].round(5)
cat_result_pred['Predicted Label'] = cat_result_pred['Predicted Label'].round(5)

print("\nModel Training Times:")
print(f"Random Forest: {np.mean(rf_train_times):.4f} seconds")
print(f"CatBoost: {np.mean(cat_train_times):.4f} seconds")
print(f"Decision Tree: {np.mean(dt_train_times):.4f} seconds")
print(f"XGBoost: {np.mean(xgb_train_times):.4f} seconds")
