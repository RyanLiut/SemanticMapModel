'''
To visualize semantic map models on the gradio
'''
import gradio as gr
import pandas as pd
import numpy as np
from SMM import SemanticMap
import os

def generate_semantic_map(file, manual_table):

    # Read from file
    if file is not None:
        if not file.name.lower().endswith(".csv"):
            raise ValueError("Only csv file is supportive")
        df = pd.read_csv(file.name)
    elif manual_table is not None and len(manual_table) > 0:
        df = pd.DataFrame(manual_table)
        df = df.replace("", pd.NA)
        df = df.dropna()

        feature_cols = df.columns[2:]
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        
        print(df)
        if df.empty:
            return None, "❌ 表格中没有有效的数据行（language 和 form 都不能为空）"
    else:
        raise ValueError("Please upload csv file of input manually!")
    
    featNames_4 = list(df.columns[2:])
    form_names = df['forms'].to_list()
    tfM_4 = df.iloc[:,2:].values
    assert len(featNames_4) == tfM_4.shape[1], print(f"Shape Mismatch between number of features {len(featNames_4)} and data {tfM_4.shape[1]}")

    '''
    tfM_4 = np.array([[0]+[1]*4+[1]*5+[0]*5+[1]*2+[0]*1, # ZH
                      [0]+[1]*2+[0]*3+[1]+[0]*7+[1]+[0]*2+[1],
                      [1]*2+[0]*5+[1]*4+[0]*2+[1,0,0]+[1,0],
                      [0]+[1]*4+[0,1]+[0]*4+[1,1]+[0]*5,

                      [1]*2+[0]*5+[1,0,1,1]+[0]*7, # TB
                      [0,0,1,1]+[0]*3+[1]+[0]*9+[1],

                      [1]*2+[0]*16, # EN
                      [1]*2+[0]*16,
                      [0]+[1]*2+[0]*3+[1]+[0]*10+[1],
                      [0]*3+[1]*3+[0,1,1]+[0]*7+[1,0],

                      [1,1]+[0]*5+[1,0,1,1]+[0,0,1]+[0]*4, # GE
                      [0]+[1]*5+[0,1,1]+[0,0,1]+[0]*3+[1,1,0],

                      [1,1]+[0]*9+[1]+[0]*6, # FR
                      [0]+[1]*4+[0]*13,

                      [1,1]+[0]*5+[1]+[0]*10, # RU
                      [0,1,1]+[0]*15,

                      [1,1]+[0]*5+[1,0,1]+[0]*8, #JP
                      [0,1,1]+[0]*15,
                      [0,0,0,1,1]+[0]*13,

                      [1,1]+[0]*5+[1]*3+[0]*8, #KO
                      [0]*4+[1]+[0]*13,
                      [0,1,1]+[0]*11+[1,0,0,1],
                      [0,0,1]+[0]*3+[1]+[0]*11,
                      [0]*3+[1]+[0,1]+[0]*12,

                      [1]+[0]*6+[1]*3+[0]*5+[1,1,0], #VI
                      [0]+[1]*3+[0]*14,
                      [0]*2+[1]*3+[0,0,1,0,1]+[0]*8,
                      [0,1,1]+[0,0,0,1]+[0]*7+[1]+[0]*3
    ])
    featNames_4 = ['类同', '补充', '重复', '延续', '更加', '减量', '还原', '条件', '任意', '极端', '严重', '无论', '上限', '顺承', '不协', '意外', '底限', '续话']#, '并列'] 
    # featNames_EN_4 = ['LT','BC', 'CF', 'YX', 'GJ', 'ZJ', 'JL', 'HY', 'TJ', 'RY', 'JD', 'YZ', 'WL', 'RB', 'SX', 'SC', 'BX', 'YW', 'DX', 'XH', 'BL']
    featNames_EN_4 = ['AF', 'SU', 'RE', 'CO', 'GD', 'DE', 'IS', 'CD', 'DC', 'PT', 'SC', 'WH', 'SE', 'SD', 'IC', 'UE', 'BL','DS']
    '''

    # GT4 = np.zeros((len(featNames_4), len(featNames_4)))
    # GT_inx = [(0,1), (0,7), (0,9), (1,2), (1,11), (1,13), (1,14), (2,3), (2,6), (2,12), (2,17), (3,4),(3,5), (3,7), (7,8), (7,9), (7,15), (8,16), (9,10)]
    # print(GT4.shape)
    # GT4[[i[0] for i in GT_inx], [i[1] for i in GT_inx]] = 1
    # GT4 += GT4.T

    savePath = f"output/smm_0.png"

    SM = SemanticMap(tfM_4, featNames_4, form_names, None)#, GT4)
    SM.get_optimal_SpanningTrees(acc_thr=0.80, figPath=savePath)
    evaluation = {k:f"{v:.2f}" for k,v in zip(["Precision", "Recall", "F1", "Degree_mean", "Degree_std"], SM.metrics)}

    eval_report = "### Evaluation Report\n"
    for key, value in evaluation.items():
        if isinstance(value, list):
            value = ", ".join(value)
        eval_report += f"- **{key}**: {value}\n"

    wrong_cases = ", ".join([SM.formNames[i] for i in SM.wrongcases])
    eval_report += f"- **Wrong Cases:** {wrong_cases}"

    return savePath, eval_report

# 保存为用户指定的文件名
def save_image(filename):
    if not filename.endswith(".png"):
        filename += ".png"
    final_path = f"output/{filename}"
    os.system(f"cp output/smm_0.png {final_path}")
    return final_path

with gr.Blocks() as demo:
    gr.Markdown("## Semantic Map Generator")

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        gr.Markdown("**or**")
        manual_table = gr.Dataframe(label="Manual Input", headers=['languages', 'forms'], row_count=5, col_count=2, type="pandas", interactive=True)
    
    preview_btn = gr.Button("Generate Preview")

    image_output = gr.Image(label="Preview Map", type="filepath")
    evaluation_output = gr.Markdown(label="Evaluation Metrics")

    with gr.Row():
        filename_input = gr.Textbox(label="Enter filename to save", placeholder="e.g., my_map.png")
        save_btn = gr.Button("Save Image")

    download_file = gr.File(label="Click to Download Saved Image")

    # 事件绑定
    preview_btn.click(fn=generate_semantic_map, inputs=[file_input,manual_table], outputs=[image_output, evaluation_output])
    save_btn.click(fn=save_image, inputs=filename_input, outputs=download_file)

demo.launch()