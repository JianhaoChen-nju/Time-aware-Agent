\section{时间感知事件搜索框架}
本章将多约束下的复杂规划问题形式化为一个在时态环境中的决策过程。为了解决现有方法在长程依赖和硬性约束上的脆弱性，本章提出了TAES框架。如图~\ref{fig:TAES}所示，该框架采用了“神经生成-符号验证”的协作范式，由三个核心模块构成：基于时态事件的统一状态空间建模、神经符号协同的集束搜索架构，以及层级化的多级状态评估机制。

\subsection{统一的时间感知事件空间}
不同于基于固定时间步长的线性规划方法，TAES将规划过程建模为在离散事件图上的导航。这种建模方式能够统一抽象粗粒度的日程规划和细粒度的动作调度，如统一抽象TimeArena 中的原子动作为原子事件与TravelPlanner中的单日行程，而非类似TimeArena中固定时间刻度每分钟，有效避免了搜索空间随物理时间跨度呈指数级爆炸的问题。

\subsubsection{事件驱动的状态定义}
TAES将智能体在时刻 $t$ 的规划状态形式化为三元组 $S_t = (\mathcal{T}_{now}, \mathcal{R}, \mathcal{C}_{met})$，其中：
\begin{itemize}
    \item $\mathcal{T}_{now} \in \mathbb{R}_{\ge 0}$ 表示当前的全局时间指针。
    \item $\mathcal{R}$ 表示当前的资源快照。在 TimeArena 中，$\mathcal{R}$ 包含空闲的厨具、实验设备或人员状态；在 TravelPlanner 中，$\mathcal{R}$ 包含剩余预算及交通工具状态。
    \item $\mathcal{C}_{met} \subseteq \mathcal{C}_{goal}$ 表示当前已满足的子目标或约束集合，例如已完成的烹饪子任务、已预订的住宿。
\end{itemize}

TAES定义事件$E$ 为搜索树上的基本转移单元。事件不仅包含动作本身如 ``chop\_tomato''，还隐含了该动作所需的持续时间 $\Delta t_E$ 和资源需求。

\subsubsection{基于事件的时间跳跃机制}
不同于传统规划模型采用等时步长的离散化方式，TAES 引入了非均匀时间步机制，即状态转移函数 $\Phi(S_t, E) \rightarrow S_{t+1}$。该函数包含两个关键逻辑步骤：

\textbf{动作执行与资源锁定}~在当前时间 $\mathcal{T}_{now}$ 尝试执行事件 $E$。系统根据时态知识图谱中的定义，锁定相关资源并更新 $\mathcal{C}_{met}$。在物理时间跳跃的区间内，被占用的资源，如切菜的智能体、已参观的景点在状态 $R$ 中被标记为锁定，当智能体在接下来的 5 分钟内切菜时，系统可以并行评估其他不依赖该智能体的动作分支，当系统规划下一天的旅游行程，可以排除已经参观的景点。

\textbf{时间快进}~这是 TAES 区别于传统规划器的核心机制。如果执行 $E$ 后系统进入资源等待状态，例如智能体正在切菜，耗时 3 分钟，系统自动将 $\mathcal{T}_{now}$ 快进到下一个决策点。决策点定义为“任一占用型资源被释放”或“任一前置依赖被满足”的时刻：
\begin{equation}
    \mathcal{T}_{next} = \min \{ \text{end\_time}(\tau) \mid \tau \in \text{RunningTasks} \}
\end{equation}
这种机制将搜索树的深度 $D$ 从物理时间长度$O(T)$压缩为关键决策次数$O(N_{actions})$，使得长程规划在计算上变得可行且高效。

\subsection{神经符号协同的集束搜索框架}
为了结合大语言模型的生成直觉与符号系统的逻辑严谨性，本章设计了基于集束搜索的协同框架。该框架维护一个宽度为 $B$的集束 $\mathcal{B}_d$，包含当前时刻最优的 $B$ 个部分计划。

\subsubsection{神经生成器}
TAES将大语言模型视为系统的思考引擎 $F_\theta$。在搜索树的每一步扩展中，LLM 接收当前状态描述 $S_t$ 和来源于 TKG 的环境上下文$C_{tkg}$，并生成 $k$ 个候选动作集合：
\begin{equation}
    \mathcal{A}_{cand} = \{E_1, E_2, ..., E_k\} \sim F_\theta(S_t, C_{tkg})
\end{equation}
在此过程中，LLM 的作用仅限于利用其常识推理能力来提供高质量的候选集，如“拿到杯子后应该去倒水而不是去洗衣服”、“去巴黎应该参观铁塔”，从而大幅缩减搜索的分支因子。其本质是一种大模型驱动的启发式搜索算法，从而避免现实世界组合爆炸的问题。

\subsubsection{搜索与剪枝流程}
\begin{algorithm}[h!]
    \caption{神经符号协同的集束搜索算法}
    \label{alg:TAES}
    \begin{algorithmic}[1] % [1] 表示显示行号
        \REQUIRE User Query $Q$, TKG $\mathcal{G}$, Beam Width $B$, Branching Factor $K$, Max Depth $D_{\text{max}}$

        \ENSURE Optimal Plan $S^*$ or Failure $\bot$
        
        \STATE \textbf{Initialize} beam $\mathcal{B}_0 \leftarrow \{ S_{\text{init}} \}$ where $S_{\text{init}} = (\mathcal{T}_{0}, \mathcal{R}_{\text{init}}, \emptyset)$
        
        \FOR{$d = 0$ \TO $D_{\text{max}}$}
            \STATE $\mathcal{C}_{\text{candidate}} \leftarrow \emptyset$ \COMMENT{Initialize candidate set for next depth}
            
            \STATE \textit{// Phase 1：大模型生成}
            \FORALL{$S \in \mathcal{B}_d$}
                \IF{\textsc{IsTerminal}($S$)}
                    \STATE Add $S$ to $\mathcal{C}_{\text{candidate}}$
                    \STATE \textbf{continue}
                \ENDIF
                
                \STATE Context $\leftarrow$ \textsc{RetrieveContext}($S, \mathcal{G}$)
                \STATE $\mathcal{A} \leftarrow$ \textsc{LLM\_Generate}($S, \text{Context}, K$)
                
                \FORALL{action $a \in \mathcal{A}$}
                    \STATE \textit{// Phase 2：事件驱动状态转移}
                    \STATE $S_{\text{new}} \leftarrow$ \textsc{ExecuteAction}($S, a$)
                    \STATE $S_{\text{next}} \leftarrow$ \textsc{TimeForward}($S_{\text{new}}$) \COMMENT{Jump to next decision point}
                    
                    \STATE \textit{// Phase 3：基于符号剪枝}
                    \IF{\textsc{SymbolicCheck}($S_{\text{next}}, \Omega_{\text{hard}}$) is \textbf{True}}
                        \STATE \textit{// Phase 4：启发式评价}
                        \STATE Score $\leftarrow$ \textsc{Evaluate}($S_{\text{next}}, V_{\text{soft}}$)
                        \STATE Add $(S_{\text{next}}, \text{Score})$ to $\mathcal{C}_{\text{candidate}}$
                    \ENDIF
                \ENDFOR
            \ENDFOR
            
            \STATE \textit{// Phase 5：筛选}
            \IF{$\mathcal{C}_{\text{candidate}} == \emptyset$}
                \RETURN $\bot$ \COMMENT{Search space exhausted (Dead end)}
            \ENDIF
            
            \STATE Sort $\mathcal{C}_{\text{candidate}}$ by Score descending
            \STATE $\mathcal{B}_{d+1} \leftarrow$ \textsc{Top-B}($\mathcal{C}_{\text{candidate}}$)
            
            \IF{All states in $\mathcal{B}_{d+1}$ are terminal}
                \STATE \textbf{break}
            \ENDIF
        \ENDFOR
        
        \RETURN $S^* \in \mathcal{B}_{\text{last}}$ with maximum Score
    \end{algorithmic}
\end{algorithm}
算法的核心是一个迭代的“扩展-评估-筛选”闭环，具体伪代码如算法\ref{alg:TAES}所示。其主要流程如下：
\begin{enumerate}
    \item \textbf{扩展：}对集束 $\mathcal{B}_d$ 中的每个父状态，调用神经生成器生成 $k$ 个子状态，总计产生 $B \times k$ 个候选路径。大语言模型在此充当直觉驱动的搜索分支引导器，它基于当前状态 $S_t$ 的描述，例如“已规划至第3天，剩余预算\$1200”及从时态知识图谱检索的上下文如附近的酒店信息，提出最具合理性的候选动作。这一步大幅压缩了暴力搜索的分支，将可能的动作空间缩减为人类常识认为合理的动作子集。
    \item \textbf{符号化评估：}调用评估函数 $V(S)$ 对所有候选路径进行打分与剪枝。$V(S)$包括两部分：通过代码逻辑强制阻断任何违反硬性约束的分支；通过启发式打分机制将风险较低的方案排序在候选的前列。
    \item \textbf{筛选：}根据分数对候选状态排序，保留前 $B$ 个最优状态进入下一轮 $\mathcal{B}_{d+1}$。这允许系统保持多条潜在的可行路径，当某条路径在未来遭遇死胡同时，其他分支能够提供有效的回溯空间。
\end{enumerate}

\subsection{多级状态评估机制}
现有的基于 LLM 的规划方法往往在生成后才进行验证，容易陷入“生成-失败-重试”的低效循环。TAES 引入了一个多级评估函数 $V(S)$，深度融合了硬逻辑剪枝与启发式引导。

\subsubsection{符号化硬约束剪枝}
在此阶段，TAES引入一组符号化规则集 $\Omega$，用于检测当前部分计划是否违反了任何即时可验证的硬性约束。如果检测到违约，状态价值立即置为 0，触发剪枝。
\begin{equation}
    V_{hard}(S) = \begin{cases} 
    0, & \text{if } \exists c \in \Omega, \text{violated}(S, c) \\ 
    1, & \text{otherwise} 
    \end{cases}
\end{equation}
具体的剪枝逻辑 $\Omega$ 包括：
\begin{itemize}
    \item \textbf{依赖性违约}：在 TimeArena 任务场景中，动作 $A$ 的执行严格依赖于其前置任务集合 $P_A$ 的完成状态。符号检查器会实时检索当前状态的 $C_{met}$ 集合，若 $P_A \not\subseteq C_{met}$，则判定为违约。在旅行规划中，这表现为逻辑链条的闭环，例如返程交通方式必须自洽，若自驾去就不能坐飞机回。
    \item \textbf{资源互斥}：在时刻 $\mathcal{T}_{now}$，系统会扫描资源快照 $\mathcal{R}$。若动作 $A$ 涉及的物理资源，如 TimeArena 中的实验设备、厨房厨具处于占用状态，或 TravelPlanner 中的餐厅处于已光顾状态，则后续重叠该资源的动作将被判定为违约。针对可消耗资源如预算，系统通过实时的数值核算进行监控。当候选动作产生的累加成本 $cost(S)$ 超过用户预设的总额时，立即进入失败状态。
    \item \textbf{属性约束}：基于用户对规划的显性要求，符号检查器会对候选属性进行校验。通过将大语言模型生成的灵活表达转化为结构化知识的表达，系统能够确保每一个被保留的规划分支在属性层面上是零违约的。通过结构化表示，符号检查器能够对涉及房型、饮食偏好及交通工具类型等等关键动作属性的严格对齐。例如，若用户要求“必须携带宠物”，则所有不具备宠物友好标签的酒店分支将被立即屏蔽。这种机制在处理多约束时，能够达到远超纯概率生成模型的稳定性 。
\end{itemize}
符号化硬约束剪枝从机制最大限度上保证了凡是通过筛选进入下一轮的计划，在逻辑上是没有冲突的。

\subsubsection{前瞻性风险评估}
通过第一阶段筛选的计划虽然逻辑上没有冲突，但可能通向死胡同，例如预算在前几天耗尽，导致后续无解。TAES引入启发式函数 $V_{soft}(S)$来实时评估当前方案的风险系数，以此估计沿着该方案扩展的未来的成功概率，引导搜索向鲁棒性更高的方向发展。$V_{soft}(S)$可以根据不同的任务针对性设计，在旅行任务中，$V_{soft}(S)$采用预算风险估算，在事件统筹任务中，$V_{soft}(S)$采用时间效率估计。下面展示两种启发式评估的具体计算方式。

\textbf{时间效率估算}~为了最小化总耗时，TAES采用关键路径估算：
\begin{equation}
    R_{time} = \frac{\text{CompletedTasks}(S)}{\mathcal{T}_{now}} + \alpha \cdot \mathbf{I}(\text{Parallelism})
\end{equation}
其中 $\mathbf{I}(\text{Parallelism})$ 是并行度指示函数。TAES会奖励那些不仅完成了任务，而且当前处于高度并行状态的路径，如人与烤箱同时工作，引导搜索向更高效的调度收敛。

\textbf{预算风险估算}~TAES提出剩余日均可用预算指标来衡量财务健康度：
\begin{equation}
    R_{budget} = \frac{Budget_{total} - \text{cost}(S)}{D_{max} - \text{depth}(S)}
\end{equation}
若 $R_{budget}$ 低于最低生存阈值 $\delta$，例如每日最低食宿成本，则给予极低分惩罚。这迫使智能体在早期做出节省预算的决策，而非贪婪地选择豪华选项。

\textbf{最终评分函数}~综合上述两阶段，最终状态评分函数定义为：
\begin{equation}
    V(S) = V_{hard}(S) \cdot [ \epsilon + (1-\epsilon) \cdot V_{soft}(S) ]
\end{equation}
其中 $\epsilon$ 是一个较小的正数，如 $10^{-1}$，确保未违约但风险高的路径优于直接违约的路径，但远劣于健康路径。这一设计使得 TAES 能够在保证零违约的前提下，自适应地寻找最优解。并且启发式评估的设计给方法的未来扩展留下了充足的空间，设计者可以根据不同的需求自定义启发式评估函数，不同的定义会引导最终得出的方案在不违反硬性约束的情况下朝着设计者希望的风格方向去发展，例如需要尽量用满预算或是尽量在伙食方面多投入预算等等