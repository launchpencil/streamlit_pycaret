import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import *
from pycaret.regression import *
from sklearn.tree import *
import plotly.figure_factory as ff
import graphviz
import matplotlib.pyplot as plt
import japanize_matplotlib
import joblib
import base64
import io

# Streamlitのページ設定
st.set_page_config(page_title="AIデータサイエンス")

# タイトルの表示
st.title("AIデータサイエンス")
st.caption("Created by Dit-Lab.(Daiki Ito)")
st.write("アップロードされたデータセットに基づいて、機械学習モデルの作成と評価を行います。")
st.write("データの読み込み　→　モデル比較　→　チューニング　→　可視化　を行うことができます")

# データファイルのアップロード
st.header("1. データファイルのアップロード")
st.caption("こちらからデータをアップロードしてください。アップロードしたデータは次のステップで前処理され、モデルの訓練に使用されます。")
uploaded_file = st.file_uploader("CSVまたはExcelファイルをアップロードしてください。", type=['csv', 'xlsx'])

# 新しいデータがアップロードされたときにセッションをリセット
if uploaded_file is not None and ('last_uploaded_file' not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_file):
    st.session_state.clear()  # セッションステートをクリア
    st.session_state.last_uploaded_file = uploaded_file.name  # 最後にアップロードされたファイル名を保存

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            train_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            train_data = pd.read_excel(uploaded_file)
        else:
            raise ValueError("無効なファイルタイプです。CSVまたはExcelファイルをアップロードしてください。")
        
        st.session_state.uploaded_data = train_data
        st.dataframe(train_data.head())
    except Exception as e:
        st.error(str(e))

# ターゲット変数の選択
if 'uploaded_data' in st.session_state:
    st.header("2. ターゲット変数の選択")
    st.caption("モデルの予測対象となるターゲット変数を選択してください。この変数がモデルの予測ターゲットとなります。")
    target_variable = st.selectbox('ターゲット変数を選択してください。', st.session_state.uploaded_data.columns)
    st.session_state.target_variable = target_variable

# 分析から除外する変数の選択
    st.header("3. 分析から除外する変数の選択")
    st.caption("モデルの訓練から除外したい変数を選択してください。これらの変数はモデルの訓練には使用されません。")
    ignore_variable = st.multiselect('分析から除外する変数を選択してください。', st.session_state.uploaded_data.columns)
    st.session_state.ignore_variable = ignore_variable

    # フィルタリングされたデータフレームの表示
    filtered_data = st.session_state.uploaded_data.drop(columns=ignore_variable)
    st.write("フィルタリングされたデータフレームの表示")
    st.write(filtered_data)

    # Excelファイルダウンロード機能
    towrite = io.BytesIO()
    downloaded_file = filtered_data.to_excel(towrite, index=False, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    original_filename = uploaded_file.name.split('.')[0]  # 元のファイル名を取得
    download_filename = f"{original_filename}_filtered.xlsx"  # フィルタリングされたファイル名を作成
    link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{download_filename}">フィルタリングされたデータフレームをExcelファイルとしてダウンロード</a>'
    st.markdown(link, unsafe_allow_html=True)

# 前処理の実行及びモデルの比較
    st.header("4. 前処理の実行とモデルの比較")
    st.caption("データの前処理を行い、利用可能な複数のモデルを比較します。最も適したモデルを選択するための基準としてください。")

    # 外れ値の処理
    remove_outliers_option = st.checkbox('外れ値を削除する', value=False) 

    if st.button('前処理とモデルの比較の実行'):  # この条件を追加
        # データの検証
        if st.session_state.uploaded_data[target_variable].isnull().any():
            st.warning("ターゲット変数に欠損値が含まれているレコードを削除します。")
            st.session_state.uploaded_data = st.session_state.uploaded_data.dropna(subset=[target_variable])
        
        # 前処理の進捗状況を表示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 前処理（モデル作成の準備）
        if 'exp_clf101_setup_done' not in st.session_state:  # このセッションで既にセットアップが完了していない場合のみ実行
            try:
                with st.spinner('データの前処理中...'):
                    exp_clf101 = setup(data=st.session_state.uploaded_data, 
                                       target=target_variable, 
                                       session_id=123, 
                                       remove_outliers=remove_outliers_option,
                                       ignore_features=ignore_variable)
                    st.session_state.exp_clf101 = exp_clf101
                    st.session_state.exp_clf101_setup_done = True  # セットアップ完了フラグをセッションに保存
                    
                    # 前処理の進捗状況を更新
                    progress_bar.progress(50)
                    status_text.text("前処理が完了しました。")
            except Exception as e:
                st.error(f"前処理中にエラーが発生しました: {str(e)}")
                st.stop()
    
        setup_list_df = pull()
        st.write("前処理の結果")
        st.caption("以下は、前処理のステップとそれに伴うデータのパラメータを示す表です。")
        st.write(setup_list_df)
    
        # モデルの比較
        try:
            with st.spinner('モデルを比較中...'):
                models_comparison = compare_models(exclude=['dummy','catboost'])
                st.session_state.models_comparison = models_comparison  # セッション状態にモデル比較を保存
                models_comparison_df = pull()
                st.session_state.models_comparison_df = models_comparison_df
                
                # モデル比較の進捗状況を更新
                progress_bar.progress(100)
                status_text.text("モデルの比較が完了しました！")
        except Exception as e:
            st.error(f"モデルの比較中にエラーが発生しました: {str(e)}")
            st.stop()

# モデルの選択とチューニング
    if 'models_comparison' in st.session_state:
        # モデル比較の表示
        models_comparison_df = pull()
        st.session_state.models_comparison_df = models_comparison_df
        st.write("モデル比較結果")
        st.caption("以下は、利用可能な各モデルの性能を示す表です。")
        st.dataframe(st.session_state.models_comparison_df)
        
        # モデル比較結果の解釈の説明を追加
        st.write("モデル比較結果の解釈:")
        st.write("- Accuracy: モデルの予測精度を示します。値が高いほどモデルの性能が良いことを示します。")
        st.write("- AUC: ROC曲線下の面積を示します。値が高いほどモデルの性能が良いことを示します。")
        st.write("- Recall: 実際の正例のうち、正しく正例と予測された割合を示します。")
        st.write("- Precision: 正例と予測されたもののうち、実際に正例である割合を示します。")
        st.write("- F1: RecallとPrecisionの調和平均を示します。両者のバランスを考慮した指標です。")
        st.write("- Kappa: モデルの予測結果と実際の結果の一致度を示します。値が高いほどモデルの性能が良いことを示します。")
        st.write("- MCC: 不均衡データにおけるモデルの性能を示します。値が高いほどモデルの性能が良いことを示します。")
        
        st.header("5. モデルの選択とチューニング")
        st.caption("最も性能の良いモデルを選択し、さらにそのモデルのパラメータをチューニングします。")
        selected_model_name = st.selectbox('使用するモデルを選択してください。', st.session_state.models_comparison_df.index)
        
        # 決定木プロットは可能なモデルのリストを表示
        tree_models = ['ada', 'et', 'rf', 'dt', 'gbr', 'catboost', 'lightgbm', 'xgboost']
        st.write("決定木プロットが可能なモデル: " + ", ".join(tree_models))

        # max_depth のオプションを表示
        if selected_model_name in tree_models:
            max_depth = st.slider("決定木の最大の深さを選択", 1, 10, 3)  # 例として最小1、最大10、デフォルト3

        # モデル名の入力
        model_name = st.text_input("保存するモデルの名前を入力してください", value="tuned_model")

        if st.button('チューニングの実行'):
            with st.spinner('チューニング中...'):
                if selected_model_name in tree_models:
                    model = create_model(selected_model_name, max_depth=max_depth)
                else:
                    # モデルの作成とチューニング
                    model = create_model(selected_model_name)
                        
                pre_tuned_scores_df = pull()
                tuned_model = tune_model(model)
                st.session_state.tuned_model = tuned_model  # tuned_modelをセッションステートに保存
                st.success("モデルのチューニングが完了しました！")
                setup_tuned_model_df = pull()
                col1, col2 = st.columns(2) 
                with col1:
                    st.write("＜チューニング前の交差検証の結果＞")
                    st.write(pre_tuned_scores_df)
                with col2:
                    st.write("＜チューニング後の交差検証の結果＞")
                    st.write(setup_tuned_model_df)
                st.caption("上記表は、チューニング前後のモデルの交差検証結果を示す表です。")
                
                # チューニング前後の比較結果の解釈の説明を追加
                st.write("チューニング前後の比較結果の解釈:")
                st.write("- チューニング後の方が、Accuracy、AUC、Recall、Precision、F1、Kappa、MCCの値が高い場合、モデルの性能が向上したことを示します。")
                st.write("- チューニング後の方が、これらの指標の値が低い場合、モデルの性能が悪化したことを示します。")
                st.write("- チューニングによる変化がない場合は、モデルの性能に大きな影響がなかったことを示します。")
                
                # チューニング後のモデルを保存
                if 'tuned_model' in st.session_state:
                    # モデルをバイナリ形式で保存
                    with open(f"{model_name}.pkl", "wb") as f:
                        joblib.dump(st.session_state.tuned_model, f)
                        
                    # ファイルをbase64エンコードしてダウンロードリンクを作成
                    with open(f"{model_name}.pkl", "rb") as f:
                        model_file = f.read()
                        model_b64 = base64.b64encode(model_file).decode()
                        href = f'<a href="data:application/octet-stream;base64,{model_b64}" download="{model_name}.pkl">チューニングされたモデルをダウンロード</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                st.header("6. モデルの可視化及び評価")
                st.caption("以下は、チューニング後のモデルのさまざまな可視化を示しています。これらの可視化は、モデルの性能や特性を理解するのに役立ちます。")
                plot_types = [
                    ('pipeline', '＜前処理パイプライン＞', '前処理の流れ（フロー）を表しています'),
                    ('residuals', '＜残差プロット＞', '実際の値と予測値との差（残差）を示しています'),
                    ('error', '＜予測誤差プロット＞', 'モデルの予測誤差を示しています'),
                    ('feature', '＜特徴量の重要度＞', '各特徴量のモデルにおける重要度を示しています'),
                    ('cooks', '＜クックの距離プロット＞', 'データポイントがモデルに与える影響を示しています'),
                    ('learning', '＜学習曲線＞', '訓練データのサイズに対するモデルの性能を示しています'),
                    ('vc', '＜検証曲線＞', 'パラメータの異なる値に対するモデルの性能を示しています'),
                    ('manifold', '＜マニホールド学習＞', '高次元データを2次元にマッピングしたものを示しています')
                ]
                for plot_type, plot_name, plot_description in plot_types:
                    try:
                        with st.spinner(f'{plot_name}のプロット中...'):
                            img = plot_model(tuned_model, plot=plot_type, display_format="streamlit", save=True)
                            st.subheader(plot_name)  # グラフのタイトルを追加
                            st.image(img)
                            st.caption(plot_description)  # グラフの説明を追加
                    except Exception as e:
                        st.warning(f"{plot_name}の表示中にエラーが発生しました: {str(e)}")
                        
                # 決定木のプロット
                if selected_model_name in tree_models:
                    st.subheader("＜決定木のプロット＞")
                    st.caption("決定木は、モデルがどのように予測を行っているかを理解するのに役立ちます。")
                    
                    try:
                        with st.spinner('決定木のプロット中...'):
                                    
                            if selected_model_name in ['dt']:
                                from sklearn.tree import plot_tree
                                fig, ax = plt.subplots(figsize=(40,20))
                                plot_tree(tuned_model, proportion=True, filled=True, rounded=True, ax=ax, max_depth=3, fontsize=14)  # フォントサイズを変更
                                st.pyplot(fig)
    
                            elif selected_model_name in ['rf', 'et']:
                                from sklearn.tree import plot_tree
                                fig, ax = plt.subplots(figsize=(40,20))
                                plot_tree(tuned_model.estimators_[0], feature_names=train_data.columns, proportion=True, filled=True, rounded=True, ax=ax, max_depth=3, fontsize=14)  # フォントサイズを変更
                                st.pyplot(fig)
                        
                            elif selected_model_name == 'ada':
                                from sklearn.tree import plot_tree
                                base_estimator = tuned_model.get_model().estimators_[0]
                                fig, ax = plt.subplots(figsize=(40,20))
                                plot_tree(base_estimator, filled=True, rounded=True, ax=ax, max_depth=3, fontsize=14)  # フォントサイズを変更
                                st.pyplot(fig)
                        
                            elif selected_model_name == 'gbr':
                                from sklearn.tree import plot_tree
                                base_estimator = tuned_model.get_model().estimators_[0][0]
                                fig, ax = plt.subplots(figsize=(40,20))
                                plot_tree(base_estimator, filled=True, rounded=True, ax=ax, max_depth=3, fontsize=14)  # フォントサイズを変更
                                st.pyplot(fig)
                        
                            elif selected_model_name == 'catboost':
                                from catboost import CatBoostClassifier, plot_tree
                                catboost_model = tuned_model.get_model()
                                fig, ax = plt.subplots(figsize=(40,20))
                                plot_tree(catboost_model, tree_idx=0, ax=ax, max_depth=3)
                                st.pyplot(fig)
                            
                            elif selected_model_name == 'lightgbm':
                                import lightgbm as lgb
                                booster = tuned_model._Booster  # LightGBM Booster object
                                fig, ax = plt.subplots(figsize=(40,20))
                                lgb.plot_tree(booster, tree_index=0, ax=ax, max_depth=3)
                                st.pyplot(fig)
                            
                            elif selected_model_name == 'xgboost':
                                import xgboost as xgb
                                booster = tuned_model.get_booster()  # XGBoost Booster object
                                fig, ax = plt.subplots(figsize=(40,20))
                                xgb.plot_tree(booster, num_trees=0, ax=ax, max_depth=3)
                                st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"決定木のプロット中にエラーが発生しました: {str(e)}") 