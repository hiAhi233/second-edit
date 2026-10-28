"""Thymoma prompt versions; values retained for existing checkpoints."""
from .base import ThreeLevelPrompts

# 胸腺瘤 CT 三级提示（前纵隔肿块）
#
# 2026-09-27 改写。旧版三层是**同一个病灶的三个名字**（anterior mediastinal mass /
# thymoma / mediastinal tumor），L3 是 L1 的同义词，完全没有体现文档「粗粒度→细粒度」
# 的多属性设计；且六句共享 "a normal chest CT" / "a chest CT with ..." 模板前缀，
# 而 BiomedCLIP 的句向量由前缀主导，换尾巴属性几乎推不动余弦。
#
# 实测（冻结编码器，_diag_prompts.py）：
#   旧版  跨层 cos(abn,abn)=0.927  cos(norm,norm)=0.945  cos(d)=0.688
#   新版  跨层 cos(abn,abn)=0.836  cos(norm,norm)=0.874  cos(d)=0.230
# 其中 d_l = t_a^l − t_n^l，异常图 pa−pn 只由它决定 —— 三个 d_l 平行则三层 map 必然
# 相同、融合权重失效。基线实测：训练前 cos(d)=0.688，训练后升到 0.984，三层 map 相关
# 0.994（单边 hinge 把 t_a 推到 −t_n，d_l≈2t_a^l，跨层 t_a 相似度直接决定 map 相同）。
#
# 新写法三层各占一个属性维度，且**句式框架互不相同**：
#   L1 器官/组织级 —— 形容词 + 模态：        a normal chest CT / an abnormal chest CT
#   L2 结构/病灶级 —— 模态 + showing + 结构： chest CT showing ...
#   L3 纹理/密度级 —— 主语前置 + on 模态：     ... density with ... margins on chest CT
THYMOMA_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest CT"],
        "2": ["chest CT showing normal mediastinal structures"],
        "3": ["homogeneous mediastinal fat density with sharp margins on chest CT"],
    },
    abnormal={
        "1": ["an abnormal chest CT"],
        "2": ["chest CT showing a soft tissue mass in the anterior mediastinum"],
        "3": ["ill-defined soft tissue density with irregular blurred margins on chest CT"],
    },
)

# 2026-09-28 加：短名词短语版。动机 —— 上面那版三层都用整句，而整句模板
# （"a chest CT with X"）会把三个差向量 d_l 压进同一个子空间：实测 cos(d)=0.230。
# 换成互不共享框架的短名词短语后 cos(d)=0.031（低一个数量级），三层天然近正交。
#
# 这条路的意义：② d_diversity_loss 存在的唯一理由就是"不显式要求分歧，三层就会塌成
# 一张图"。而实测②正是打垮单层的元凶（只开②，L1 的 Dice 0.6215→0.4742）。**如果提示词
# 本身就给出正交的 d，就不需要②了** —— 既拿到分歧，又不用付单层变差的代价。
#
# 保留方式：不覆盖上面那份，改用 --prompt-set 切换。因为 thymoma_local.pt 是用上面那版
# 训的，直接改掉会让标定/出图配错词表（改动 8 附录记过这个坑）。
THYMOMA_PROMPTS_SHORT = ThreeLevelPrompts(
    normal={
        "1": ["normal mediastinal tissue"],
        "2": ["normal thymus"],
        "3": ["homogeneous mediastinal fat"],
    },
    abnormal={
        "1": ["mediastinal mass"],
        "2": ["thymoma"],
        "3": ["ill-defined soft tissue"],
    },
)

# 2026-09-28 加：分层版（外部建议稿，global / attribute×4 / lesion 三个层级）。
# 我们的框架是 N 层通用（层数由 prompts.levels 决定、融合权重按层数建），而这份设计里
# 属性层有 4 个属性 —— **不能把 4 个属性平均成一层**：实测平均后 cos(d) 会从 0.128 涨到
# 0.479（正交分量互相抵消，剩下的共同分量正好撞上 L1/L3），比现用的 0.230 还差。
# 所以按 4 个属性各自成层装，共 6 层。
#
# 实测 6x6 cos(d)：均值 +0.128（优于现用 +0.230），最大 +0.785（差于现用 +0.481）。
# **已知缺陷：L1 全局 ↔ L2 结构 = +0.785** —— 'an abnormal chest CT' 与
# 'abnormal mediastinal contours' 近乎同义，这两层会出几乎一样的图。其余属性对都很好
# （L4 边缘与所有层近正交）。先按原稿跑，再决定要不要删掉冗余的那层。
THYMOMA_PROMPTS_TIERS = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest CT"],                                              # 全局
        "2": ["chest CT with normal mediastinal structures"],                    # 属性·结构
        "3": ["chest CT with homogeneous mediastinal fat density"],              # 属性·密度纹理
        "4": ["chest CT with sharp and well-defined margins"],                   # 属性·边缘
        "5": ["chest CT with bilaterally symmetric structures"],                 # 属性·对称性
        "6": ["normal mediastinal tissue"],                                      # 病灶
    },
    abnormal={
        "1": ["an abnormal chest CT"],
        "2": ["chest CT with abnormal mediastinal contours"],
        "3": ["chest CT with heterogeneous soft tissue density"],
        "4": ["chest CT with ill-defined and blurred margins"],
        "5": ["chest CT with asymmetric mediastinal structures"],
        "6": [
            "a soft tissue mass in the anterior mediastinum",
            "irregular and blurred margins",
            "a lesion with a soft tissue mass",
            "a mass with mass effect on adjacent structures",
            "a heterogeneous enhancing lesion",
        ],
    },
)

# 2026-09-28 加：分层设计的**原版**（3 层）。上面 THYMOMA_PROMPTS_TIERS 把 4 个属性拆成了
# 4 层，那是误读 —— 原设计里 4 个属性同属 L2，作为一层的多条提示词（框架内 _mean_embed
# 会把同一层的多条做均值池化）。三层各自成层，层数仍是 3，与既有机制完全对齐。
#
# 训练前实测 cos(d)：均值 +0.479、最大 +0.557（L1-L2）。对照：现用整句 3 层 +0.230、
# 短名词短语 3 层 +0.031、6 层拆法 +0.128/+0.785。
# 注意：把 4 个属性池化后，它们各自近正交的分量（两两均值 −0.05）会互相抵消，剩下的
# 共同分量反而更贴近 L1/L3 —— 所以三层装法的 cos(d) 比 6 层拆法**更高**，不是更低。
THYMOMA_PROMPTS_TIERS3 = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest CT"],
        "2": [                                                    # 属性层：4 个属性池化
            "chest CT with normal mediastinal structures",
            "chest CT with homogeneous mediastinal fat density",
            "chest CT with sharp and well-defined margins",
            "chest CT with bilaterally symmetric structures",
        ],
        "3": ["normal mediastinal tissue"],
    },
    abnormal={
        "1": ["an abnormal chest CT"],
        "2": [
            "chest CT with abnormal mediastinal contours",
            "chest CT with heterogeneous soft tissue density",
            "chest CT with ill-defined and blurred margins",
            "chest CT with asymmetric mediastinal structures",
        ],
        "3": [
            "a soft tissue mass in the anterior mediastinum",
            "irregular and blurred margins",
            "a lesion with a soft tissue mass",
            "a mass with mass effect on adjacent structures",
            "a heterogeneous enhancing lesion",
        ],
    },
)

# 2026-09-28 加：把 L2（属性）与 L3（病灶）合并成一层 → 2 层设计（L1 全局 + L2' 合并层）。
# 起因是"觉得 L2 和 L3 两个维度蛮接近"。但实测最接近的一对其实是 **L1-L2**（cos(d)=+0.556），
# L2-L3 只有 +0.478 —— 合并 L2+L3 反而把最冗余的 L1-L2 留在了场上。仍按需求跑。
#
# 裁剪（_mean_embed 是均值池化，条数比例即权重），两处删除：
#   1. 删属性"结构"那条（'chest CT with normal mediastinal structures' 等）。它是实测最冗余
#      的一条 —— 6 层分析里 L1(全局)-结构 = +0.785，是所有配对里最平行的。既然合并就是
#      为了去冗余，留着它等于把冗余又搬进来。删后与 L1 的分歧从 cos(d,L1)=+0.498
#      降到 +0.368，正是想要的。代价是离纯 L2 远了些（保留 L2 方向 0.771→0.690）。
#   2. 病灶留 3 条（位置形态 / 占位效应 / 信号密度），去掉 'a lesion with a soft tissue mass'
#      （与第 1 条重复）与 'irregular and blurred margins'（是片段不是病灶名，且"边缘"已由
#      属性层覆盖）。
# 实测：同层 cos(n,a)=+0.914，cos(d_L1-L2')=+0.368，保留 L2 方向 0.690。
THYMOMA_PROMPTS_MERGE23 = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest CT"],
        "2": [
            # 属性（原 L2，删掉"结构"那条，见上）
            "chest CT with homogeneous mediastinal fat density",
            "chest CT with sharp and well-defined margins",
            "chest CT with bilaterally symmetric structures",
            # 病灶（原 L3，1 条）
            "normal mediastinal tissue",
        ],
    },
    abnormal={
        "1": ["an abnormal chest CT"],
        "2": [
            # 属性（原 L2，删掉"结构"那条）
            "chest CT with heterogeneous soft tissue density",
            "chest CT with ill-defined and blurred margins",
            "chest CT with asymmetric mediastinal structures",
            # 病灶（原 L3，3 条：位置形态 / 占位效应 / 信号密度）
            "a soft tissue mass in the anterior mediastinum",
            "a mass with mass effect on adjacent structures",
            "a heterogeneous enhancing lesion",
        ],
    },
)

# 2026-09-28 加：单层设计 —— 只有"属性池化"这一层，没有 L1 全局、没有 L3 病灶、没有融合。
#
# 依据：在 tiers3 设计内丢层是唯一让结果涨回来的操作（三层 0.6224 → L1+L2 0.6259 →
# 只留 L2 0.6265，改动 15 第 (3) 条）。但那个 0.6265 是**事后从三层模型里挑最好的层**
# （用测试集选的），不是训练时就只有这一层。本跑把这件事做干净：
# 训练时就是单一锚点对，`d_diversity_loss` 自动退化为 0（len(ds)<2），融合权重恒为 1。
#
# 要回答的问题：**"多属性提示词池化成单维锚点"这个做法本身是否有效、且一层就够？**
# 若是，课题的卖点可以从"三层融合"改成"属性池化锚点"——更简洁，且与基线的差距消失。
THYMOMA_PROMPTS_ATTRONLY = ThreeLevelPrompts(
    normal={
        "1": [
            "chest CT with normal mediastinal structures",
            "chest CT with homogeneous mediastinal fat density",
            "chest CT with sharp and well-defined margins",
            "chest CT with bilaterally symmetric structures",
        ],
    },
    abnormal={
        "1": [
            "chest CT with abnormal mediastinal contours",
            "chest CT with heterogeneous soft tissue density",
            "chest CT with ill-defined and blurred margins",
            "chest CT with asymmetric mediastinal structures",
        ],
    },
)

PROMPT_SETS = {
    "sentence": THYMOMA_PROMPTS,        # 改动 7 的整句版，现默认，基线 ckpt 用它训的
    "short": THYMOMA_PROMPTS_SHORT,     # 短名词短语版，cos(d)=0.031
    "tiers": THYMOMA_PROMPTS_TIERS,     # 6 层版（我误把 4 个属性拆成 4 层，非原设计）
    "tiers3": THYMOMA_PROMPTS_TIERS3,   # 原设计：L1 全局 / L2 属性(4 条池化) / L3 病灶
    "merge23": THYMOMA_PROMPTS_MERGE23, # L2+L3 合并成一层 → 2 层
    "attronly": THYMOMA_PROMPTS_ATTRONLY,  # 单层：只有 4 条属性池化
}


THYMOMA_SENTENCE_PROMPTS = THYMOMA_PROMPTS
