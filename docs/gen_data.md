Dưới đây là bản tổng hợp chi tiết toàn bộ kế hoạch thiết kế bộ sinh dữ liệu (Data Generator Plan) cho bài toán FJSP trong nhà máy bán dẫn của bạn. Kế hoạch này được cấu trúc để bạn có thể code thành một script độc lập, truyền vào một `random_seed` và xuất ra một bộ dataset hoàn chỉnh dưới dạng file JSON.

---

### PHẦN 1: MÔ TẢ BÀI TOÁN & GIẢ ĐỊNH (Problem Definition & Assumptions)

* **Môi trường:** Nhà máy sản xuất bán dẫn (Semiconductor Fab).
* 
**Loại hình:** High-Volume / Low-Mix (HV/LM). Số lượng sản phẩm ít (ví dụ: 2 loại sản phẩm chính) nhưng số lượng lot (đơn hàng) cực kỳ lớn.


* **Tính chất lập lịch:** Static Scheduling (Lập lịch tĩnh). Toàn bộ các đơn hàng (Jobs) đã sẵn sàng tại thời điểm $t=0$ ở Kho trung tâm (Stockroom).
* **Mục tiêu tối ưu:** Cực tiểu hóa Makespan (Thời gian hoàn thành tất cả job), Min Total Weighted Tardiness (Tổng trễ hạn có trọng số, ưu tiên Hot Lots), và Min Setup Cost (Thời gian/Chi phí chuyển đổi máy).
* 
**Các giả định nới lỏng tạm thời:** Không có lò khuếch tán (No batching/Diffusion) để giảm độ phức tạp, và sức chứa của hệ thống đệm (Buffer capacity) tại các máy là vô hạn.



---

### PHẦN 2: CẤU TRÚC LƯU TRỮ DỮ LIỆU (JSON Data Schema)

Dữ liệu đầu ra sẽ được gom vào một tệp JSON duy nhất gồm 4 block chính: `factory_infrastructure`, `transport_matrices`, `product_recipes`, và `job_list`.

#### 2.1. Cấu trúc Hạ tầng Nhà máy (`factory_infrastructure`)

Tổ chức theo phân cấp 3 tầng: **Area $\rightarrow$ Workstation Group (WSG) $\rightarrow$ Workstation (WS) $\rightarrow$ Tool**.

* **Logic Generate:**
* Định nghĩa danh sách các **Area** (ví dụ: `LITH`, `ETCH`, `IMPL`, `FILM`, `METL`).


* Bên trong mỗi Area, tạo 1-3 **WSG** (nhóm công nghệ, ví dụ: `LITH_STEPPER`, `LITH_SCANNER`).
* Bên trong mỗi WSG, chia làm 1-2 **WS** để tạo sự chuyên biệt (Dedication) cho Sản phẩm A hoặc B.
* Bên trong mỗi WS, sinh ra $N$ **Tools** (các máy vật lý cụ thể).


* **Khởi tạo Trạng thái ($t=0$):** Quét qua toàn bộ Tools. Random chọn một công thức (Recipe) gán vào trường `initial_setup_state` để làm cơ sở tính chi phí setup cho lot đầu tiên chạy vào máy.

#### 2.2. Dữ liệu Vận chuyển AMHS (`transport_matrices`)

* **Logic Generate:**
* Tạo ra một ma trận vuông cố định (cấp độ Area $\rightarrow$ Area và WS $\rightarrow$ WS).
* Với mỗi cặp điểm $(i, j)$, sinh một giá trị thời gian di chuyển cơ sở ngẫu nhiên trong một dải cho trước.
* *Quy tắc:*
* Kho (Stockroom) đến các Area: Random trong khoảng 10 - 20 phút.
* Khác Area (Inter-Area): Random trong khoảng 15 - 30 phút.
* Cùng Area, khác WS (Inter-WS): Random trong khoảng 5 - 10 phút.
* Cùng WS (Intra-WS): 1 - 3 phút.



#### 2.3. Lộ trình & Ràng buộc Công nghệ (`product_recipes`)

Vì là HV/LM, bạn sẽ tạo 2 danh sách lộ trình (Routes) cho `Product_A` và `Product_B` (số lượng product thực tế tôi có thể config).

* **Logic Generate Lộ trình (Routing):**
* Lặp từ bước 1 đến $M$ (với $M$ có thể từ 50-100 bước để đơn giản hóa lúc đầu, nhưng phải có tính lặp lại/re-entrant).
* Mỗi bước (Step) quy định: `Step_ID`, `Target_WSG` (Nhóm máy bắt buộc), và `Nominal_Processing_Time` (Thời gian xử lý danh định, tính bằng phút).


* **Logic Generate Ràng buộc Setup (Setup Matrix):**
* Tạo ma trận thời gian chuyển đổi (Sequence-Dependent Setup) cho các WSG đặc thù.
* 
*Khu vực LITH (Quang khắc):* Bất cứ sự thay đổi Recipe (đổi mặt nạ) nào đều mất từ 5 - 20 phút.


* 
*Khu vực IMPL (Cấy Ion) & ETCH (Khắc khô):* Thay đổi nhóm khí gas/hóa chất mất 10 - 15 phút.


* Nếu hai job liên tiếp cùng Recipe $\rightarrow$ Setup Time $= 0$.



#### 2.4. Danh sách Đơn hàng tại $t=0$ (`job_list`)

Đây là danh sách $N$ lot đổ vào nhà máy cùng một lúc từ Stockroom.

* **Logic Generate:**
* Vòng lặp tạo $N$ jobs.
* **Product Type:** Chọn ngẫu nhiên `Product_A` hoặc `Product_B`.
* **Priority:** Gieo xác suất. 97.5% là `Normal`, 2.5% là `Hot_Lot` (đơn hàng khẩn cấp).


* **Hạn chót (Due Date - $D_i$):** Tính bằng công thức $D_i = \text{Total\_Nominal\_Processing\_Time} \times \alpha$.
* Với kịch bản ngặt nghèo (High tightness): Random $\alpha \in [1, 2]$.


* Với kịch bản nới lỏng (Low tightness): Random $\alpha \in [1, 3]$.




* **Vị trí ban đầu:** Tất cả được gán giá trị `current_location = "Central_Stockroom"`.



---

### TÓM TẮT LUỒNG THỰC THI (Execution Flow cho File Script)

Bạn chỉ cần thiết kế một file Python chạy tuần tự theo các hàm sau:

1. `set_seed(42)`
2. `factory = generate_factory_infrastructure()`
3. `amhs_matrix = generate_transport_matrix(factory)`
4. `recipes, setup_matrices = generate_product_routes()`
5. `jobs = generate_job_list(num_jobs, recipes)`
6. `save_to_json(factory, amhs_matrix, recipes, setup_matrices, jobs, "fjsp_dataset_seed42.json")`

Với cách thiết kế này, dữ liệu của bạn hoàn toàn tĩnh (static) cho mỗi lần chạy mô phỏng hoặc train RL, giúp bạn dễ dàng debug, tính toán baseline (dùng các thuật toán heuristic kinh điển để so sánh với RL) và lưu trữ lại kết quả để phân tích.

Khi định nghĩa `State` (trạng thái đầu vào) cho Agent RL quan sát từ bộ dữ liệu này, bạn dự định sẽ cho Agent nhìn thấy toàn bộ thông tin của nhà máy (Centralized Observation), hay mỗi Agent tại một trạm chỉ nhìn thấy hàng đợi và các máy xung quanh nó (Decentralized/Multi-Agent)?