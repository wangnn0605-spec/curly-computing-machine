import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# -----------------------------------------------------------------------------
# 強制設定 Matplotlib 使用繁體中文（微軟正黑體）並正常顯示負號
# -----------------------------------------------------------------------------
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False 

# -----------------------------------------------------------------------------
# 頁面基本設定
# -----------------------------------------------------------------------------
st.set_page_config(page_title="平面桁架靜力學分析教學系統", layout="wide")
st.title("🏗️ 平面桁架靜力學分析輔助教學應用程式")
st.write("本系統支援自動辨識零力桿、計算桿件內力，並以顏色區分拉力（紅色）與壓力（藍色）。")

# -----------------------------------------------------------------------------
# 側邊欄：互動式參數輸入
# -----------------------------------------------------------------------------
st.sidebar.header("📌 1. 輸入桁架幾何與外力")

example = st.sidebar.selectbox("選擇輸入模式", ["使用預設經典範例", "自定義輸入"])

if example == "使用預設經典範例":
    nodes_input = "0,0\n4,0\n2,3"
    elements_input = "0,1\n1,2\n2,0"
    supports_input = "0, UX, UY\n1, UY" 
    loads_input = "2, 0, -10" 
else:
    nodes_input = st.sidebar.text_area("節點座標 (X, Y) 每行一個", "0,0\n3,0\n6,0\n6,4\n3,4\n0,4")
    elements_input = st.sidebar.text_area("桿件連接 (節點i, 節點j) 每行一個", "0,1\n1,2\n2,3\n3,4\n4,5\n5,0\n0,4\n1,4\n2,4")
    supports_input = str(st.sidebar.text_area("支承條件 (節點, 方向限制) e.g., 0, UX, UY", "0, UX, UY\n2, UY"))
    loads_input = st.sidebar.text_area("外加負載 (節點, Fx, Fy) 每行一個", "5, 9, 0\n1, 0, -15")

# -----------------------------------------------------------------------------
# 資料解析與核心計算
# -----------------------------------------------------------------------------
try:
    # 1. 解析節點
    nodes = []
    for line in nodes_input.strip().split('\n'):
        if line: nodes.append([float(x) for x in line.split(',')])
    nodes = np.array(nodes)
    num_nodes = len(nodes)

    # 2. 解析桿件
    elements = []
    for line in elements_input.strip().split('\n'):
        if line: elements.append([int(x) for x in line.split(',')])
    num_elements = len(elements)

    # 3. 初始化剛度矩陣與負載向量
    K_global = np.zeros((2 * num_nodes, 2 * num_nodes))
    F_global = np.zeros(2 * num_nodes)

    # 4. 解析外力
    for line in loads_input.strip().split('\n'):
        if line:
            parts = line.split(',')
            n_idx = int(parts[0])
            fx = float(parts[1])
            fy = float(parts[2])
            F_global[2 * n_idx] = fx
            F_global[2 * n_idx + 1] = fy

    # 5. 組裝結構剛度矩陣 (假設所有桿件 EA = 10000)
    EA = 10000.0
    for i, elem in enumerate(elements):
        n1, n2 = elem
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.hypot(x2 - x1, y2 - y1)
        cx = (x2 - x1) / L
        cy = (y2 - y1) / L
        
        k_local = (EA / L) * np.array([
            [cx*cx,  cx*cy, -cx*cx, -cx*cy],
            [cx*cy,  cy*cy, -cx*cy, -cy*cy],
            [-cx*cx, -cx*cy, cx*cx,  cx*cy],
            [-cx*cy, -cy*cy, cx*cy,  cy*cy]
        ])
        
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for r in range(4):
            for c in range(4):
                K_global[dofs[r], dofs[c]] += k_local[r, c]

    # 6. 處理支承邊邊界條件
    restrained_dofs = []
    for line in supports_input.strip().split('\n'):
        if line:
            parts = line.split(',')
            n_idx = int(parts[0])
            for part in parts[1:]:
                cond = part.strip().upper()
                if 'UX' in cond: restrained_dofs.append(2 * n_idx)
                if 'UY' in cond: restrained_dofs.append(2 * n_idx + 1)

    active_dofs = [i for i in range(2 * num_nodes) if i not in restrained_dofs]

    # 7. 求解節點位移
    U_global = np.zeros(2 * num_nodes)
    K_sub = K_global[np.ix_(active_dofs, active_dofs)]
    F_sub = F_global[active_dofs]
    
    U_sub = np.linalg.solve(K_sub, F_sub)
    U_global[active_dofs] = U_sub

    # 8. 計算各桿件內力
    element_forces = []
    for elem in elements:
        n1, n2 = elem
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.hypot(x2 - x1, y2 - y1)
        cx = (x2 - x1) / L
        cy = (y2 - y1) / L
        
        u = U_global[[2*n1, 2*n1+1, 2*n2, 2*n2+1]]
        force = (EA / L) * np.dot(np.array([-cx, -cy, cx, cy]), u)
        element_forces.append(force)

# -----------------------------------------------------------------------------
# 前端呈現與視覺化
# -----------------------------------------------------------------------------
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 2. 逐步計算與內力分析結果")
        
        results_data = []
        for idx, force in enumerate(element_forces):
            if abs(force) < 1e-4:
                status = "零力桿 (Zero-force)"
                color_label = "⚫ 黑色"
                force_val = 0.0
            elif force > 0:
                status = "張力 (Tension)"
                color_label = "🔴 紅色"
                force_val = round(force, 3)
            else:
                status = "壓力 (Compression)"
                color_label = "🔵 藍色"
                force_val = round(abs(force), 3)
                
            results_data.append({
                "桿件編號": f"桿件 {idx} (節點 {elements[idx][0]}-{elements[idx][1]})",
                "內力大小 (kN)": force_val,
                "受力型態": status,
                "顯示顏色": color_label
            })
            
        st.dataframe(results_data, use_container_width=True)

        st.info("""
        💡 **教學小筆記 (靜力學對照技巧)：**
        * **張力 (Tension)**：算出來為正值，右圖會以**紅色**實線顯示。
        * **壓力 (Compression)**：算出來為負值，右圖會以**藍色**實線顯示。
        * **零力桿 (Zero-Force)**：符合特殊節點規則者，會以**黑色虛線**表示！
        """)

    with col2:
        st.subheader("🎨 3. 桁架內力視覺化圖形")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # 繪製各個桿件
        for idx, elem in enumerate(elements):
            n1, n2 = elem
            x = [nodes[n1][0], nodes[n2][0]]
            y = [nodes[n1][1], nodes[n2][1]]
            
            force = element_forces[idx]
            if abs(force) < 1e-4:
                color = 'black'
                linestyle = '--'
                linewidth = 1.5
            elif force > 0:
                color = 'red'
                linestyle = '-'
                linewidth = 2.5
            else:
                color = 'blue'
                linestyle = '-'
                linewidth = 2.5
                
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, zorder=1)
            
            # 桿件中心點標註力的大小
            mx, my = np.mean(x), np.mean(y)
            ax.text(mx, my, f"{abs(force):.1f}", color='darkgreen', fontsize=10, 
                    weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 繪製節點
        ax.scatter(nodes[:, 0], nodes[:, 1], color='gray', s=150, zorder=2)
        for i, node in enumerate(nodes):
            ax.text(node[0], node[1] + 0.15, f"N{i}", fontsize=12, color='black', weight='bold', ha='center')

        # 改回全中文標籤
        ax.set_xlabel("X 座標 (公尺)")
        ax.set_ylabel("Y 座標 (公尺)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_axisbelow(True)
        
        # 改回全中文圖例
        ax.plot([], [], color='red', label='張力 (紅色實線)', linewidth=2.5)
        ax.plot([], [], color='blue', label='壓力 (藍色實線)', linewidth=2.5)
        ax.plot([], [], color='black', linestyle='--', label='零力桿 (黑色虛線)', linewidth=1.5)
        ax.legend(loc='upper right')
        
        st.pyplot(fig)

except Exception as e:
    st.error(f"輸入資料格式有誤，請依範例格式檢查。錯誤訊息: {e}")

