Thiết kế hàm thưởng (reward function) là một trong những yếu tố quyết định thành bại khi áp dụng Học tăng cường sâu (Deep Reinforcement Learning - DRL) vào bài toán lập kế hoạch sản xuất và điều độ (scheduling problem, như Job Shop Scheduling Problem - JSSP, Flexible Job Shop Scheduling - FJSSP).

Trong bài toán lập lịch, mô hình DRL thường đưa ra quyết định theo hướng **xây dựng lịch trình từng bước** (Constructive MDP) hoặc **tối ưu hóa/cải thiện lịch trình sẵn có** (Improvement MDP). Dưới đây là hướng dẫn chi tiết và hệ thống về cách thiết kế hàm thưởng cho bài toán này dưới góc độ lý thuyết và thực thi thực tế.

### 1. Phân loại hàm thưởng: Sparse Reward vs. Dense Reward

---

Khi thiết kế reward, thử thách lớn nhất của bài toán lập lịch là **tính hoãn lại của kết quả** (delayed feedback): một quyết định phân công máy ở thời điểm ban đầu có thể ảnh hưởng lớn đến tổng thời gian hoàn thành (Makespan) ở cuối cùng.

#### a. Thưởng thưa (Sparse / Terminal Reward)

Mô hình chỉ nhận phản hồi ở bước cuối cùng khi toàn bộ các công việc (jobs/operations) đã được xếp lịch hoàn tất.

* **Công thức tổng quát**:

$$r_t = \begin{cases} -C_{\max} & \text{nếu } t = T \text{ (hoàn thành lịch thanh toán/xếp lịch)} \\ 0 & \text{nếu } t < T \end{cases}$$



hoặc sử dụng nghịch đảo/tỷ lệ: $r_T = \frac{C_{\text{heuristic}}}{C_{\max}}$.
* **Ưu điểm**: Đảm bảo đúng mục tiêu tối ưu toàn cục, không làm lệch hướng học của tác tử (agent).
* **Nhược điểm**: Hiện tượng gán tín hiệu thưởng (credit assignment problem) rất khó khăn. Tác tử rất khó học trong không gian trạng thái lớn vì đa số các bước đi $t < T$ đều trả về 0.

#### b. Thưởng dày (Dense / Intermediate Reward)

Tác tử nhận thưởng hoặc phạt ngay tại mỗi bước ra quyết định $t$.

* **Công thức chênh lệch thời gian hoàn thành (Delta Makespan)**:

$$r_t = -(C_{\max}(s_{t+1}) - C_{\max}(s_t))$$



Trong đó $C_{\max}(s_t)$ là thời gian hoàn thành ước tính hoặc thực tế của hệ thống tại trạng thái $s_t$.
* **Ưu điểm**: Cung cấp phản hồi liên tục, giúp mạng thần kinh hội tụ nhanh hơn rất nhiều.
* **Nhược điểm**: Dễ dẫn đến hiện tượng cận thị (myopic decisions) hoặc tối ưu cục bộ nếu chỉ số ước tính từng bước không phản ánh đúng bức tranh tổng thể.

### 2. Thiết kế hàm thưởng theo chỉ số mục tiêu (Performance Metrics)

---

Mỗi bài toán điều độ sẽ có mục tiêu tối ưu khác nhau. Dưới đây là cách cụ thể hóa reward cho từng mục tiêu chính:

#### a. Tối thiểu hóa tổng thời gian hoàn thành (Makespan - $C_{\max}$)

Makespan là thời điểm máy cuối cùng hoàn thành xong công việc cuối cùng.

* **Cách 1: Phạt thời gian chờ/thời gian rảnh (Idle Time Penalty)**

$$r_t = - \sum_{m \in M} \text{IdleTime}(m, t)$$



Phạt agent dựa trên tổng thời gian máy ngồi không (idle) sau khi gán thao tác mới.
* **Cách 2: Giảm giới hạn dưới ước tính (Estimated Lower Bound Reduction)**
Tại trạng thái $s_t$, gọi $LB(s_t)$ là giới hạn dưới của Makespan dựa trên đường găng (critical path). Reward tại bước $t$:

$$r_t = -(LB(s_{t+1}) - LB(s_t))$$



Nếu hành động $a_t$ làm tăng độ dài đường găng dự kiến, agent sẽ bị phạt lập tức.

#### b. Tối thiểu hóa độ trễ hạn chót (Total / Maximum Tardiness - $T_{\max}$)

Trong sản xuất thực tế, mỗi công việc $j$ có thời hạn giao hàng (Due Date - $d_j$) và thời gian hoàn thành $C_j$. Độ trễ $T_j = \max(0, C_j - d_j)$.

* **Reward từng bước khi hoàn thành 1 công việc $j$**:

$$r_t = -\max(0, C_j - d_j)$$


* **Reward tích lũy từng bước thời gian**:

$$r_t = -\sum_{j \in J_{\text{active}}} \max(0, t - d_j)$$



Hàm này phạt agent liên tục cho mỗi đơn vị thời gian mà các công việc đang trong hệ thống bị quá hạn.

#### c. Tối đa hóa hiệu suất sử dụng máy (Machine Utilization)

$$r_t = \frac{\sum_{m \in M} \text{WorkingTime}(m, t)}{\vert{}M\vert{} \times \Delta t}$$


Nên kết hợp chỉ số này với Makespan để tránh việc cố tình kéo dài thời gian gia công chỉ để tăng tỉ lệ sử dụng máy.

### 3. Biến đổi hàm thưởng nâng cao: Potential-Based Reward Shaping (PBRS)

---

Để vừa đảm bảo tính hội tụ nhanh của Dense Reward, vừa giữ nguyên chính sách tối ưu toàn cục của Sparse Reward, phương pháp **Shaping thưởng dựa trên tiềm năng (Potential-Based Reward Shaping - PBRS)** là tiêu chuẩn vàng trong lý thuyết DRL.

Hàm thưởng được định nghĩa:


$$F(s_t, s_{t+1}) = \gamma \Phi(s_{t+1}) - \Phi(s_t)$$


Trong đó:

* $\Phi(s)$ là **hàm tiềm năng** (Potential Function) đánh giá "độ tốt" của trạng thái $s$.
* $\gamma \in [0, 1)$ là hệ số chiết khấu (discount factor).

**Ứng dụng trong Scheduling:**
Bạn có thể chọn $\Phi(s) = -LB_{\text{makespan}}(s)$ (âm của thời gian hoàn thành ước tính ngắn nhất từ trạng thái $s$).


$$r'_t = r_t^{\text{gốc}} + \gamma \Phi(s_{t+1}) - \Phi(s_t)$$


PBRS đảm bảo mặt toán học rằng chính sách tối ưu (optimal policy $\pi^*$) của MDP sẽ không bị thay đổi so với khi dùng hàm thưởng ban đầu.

### 4. Xử lý bài toán đa mục tiêu (Multi-Objective Scheduling)

---

Thực tế sản xuất thường yêu cầu cân bằng giữa Makespan, Chi phí năng lượng (Energy Consumption), và Độ trễ (Tardiness).

#### Phương pháp Tổng chi phí có trọng số (Weighted Sum Approach)

$$R_t = w_1 \cdot R_{\text{makespan}} + w_2 \cdot R_{\text{tardiness}} + w_3 \cdot R_{\text{energy}}$$


Trong đó $\sum w_i = 1$.

**Lưu ý quan trọng khi phối hợp:**

1. **Chuẩn hóa quy mô (Normalization):** Các giá trị như thời gian (giờ) và năng lượng (kWh) có đơn vị khác nhau. Bắt buộc phải đưa về cùng khoảng giá trị (ví dụ $[-1, 0]$ hoặc $[-1, 1]$) trước khi nhân trọng số $w_i$.

$$R_{\text{norm}} = \frac{R - R_{\min}}{R_{\max} - R_{\min}}$$


2. **Dynamic Weighting (Trọng số động):** Thay đổi trọng số $w_i$ theo thời gian học hoặc trạng thái hệ thống (ví dụ: nếu phát hiện trễ hạn chót gia tăng, tự động nâng $w_2$).

### 5. Cạm bẫy thường gặp & Kinh nghiệm thực tế (Best Practices)

---

1. **Tránh Reward Hacking (Tác tử "lách" quy tắc):**
* *Ví dụ:* Nếu chỉ phạt hành động vi phạm ràng buộc (invalid action) bằng điểm phạt rất nặng (ví dụ $-1000$), agent có thể trở nên quá sợ hãi và chỉ chọn các hành động an toàn nhưng hiệu quả rất kém.
* *Giải pháp:* Thay vì phạt nặng hành động không hợp lệ, hãy sử dụng kỹ thuật **Action Masking** (che biến các hành động không hợp lệ ở tầng output của mạng thần kinh).


2. **Kỹ thuật Clip và Scale Reward:**
* Các thuật toán như PPO (Proximal Policy Optimization) hoặc DQN (Deep Q-Network) rất nhạy cảm với biên độ lớn của Reward.
* Nên căn chỉnh sao cho giá trị Reward nằm trong khoảng $[-1, 1]$ hoặc $[0, 1]$. Việc chia cho giá trị Makespan ước tính ban đầu từ thuật toán heuristic (như FIFO, MWKR) là một cách chuẩn hóa hiệu quả:

$$r_t = \frac{C_{\text{heuristic}} - C_{\text{DRL}}}{C_{\text{heuristic}}}$$




3. **Chiến lược khuyến nghị cho bạn khi bắt đầu thử nghiệm:**
* **Bước 1:** Bắt đầu bằng **Action Masking** để loại bỏ hoàn toàn các quyết định vi phạm ràng buộc.
* **Bước 2:** Sử dụng **Dense Reward dựa trên Delta Estimated Makespan** ($r_t = -(LB(s_{t+1}) - LB(s_t))$).
* **Bước 3:** Chuẩn hóa Reward về khoảng $[-1, 0]$.
* **Bước 4:** Nếu bài toán phức tạp, áp dụng **Potential-Based Reward Shaping (PBRS)** với tiềm năng lấy từ đường găng (critical path) của đồ thị công việc (Disjunctive Graph).