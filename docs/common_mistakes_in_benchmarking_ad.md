# Common Mistakes in Benchmarking Autonomous Driving

> "Take nobody's word for it."
> 
> &mdash; <cite>Motto of the Royal Society</cite>

Evaluating entire autonomous driving systems is a complex task, that is prone to subtle but significant errors. Unfortunately, such methodological mistakes are widespread in the autonomous driving literature.
This document aims to outline these issues so that reviewers can identify them and authors can avoid them.
The intent of this document is not to criticize individual authors or works. We all make mistakes. 
Science is a self-correcting process, what matters is how we address and rectify issues.
As such, we will focus on the methodological mistakes, without citing the works we refer to.

Many of the points we raise may be familiar to experts, but are often stated implicitly rather than explicitly. As a result, readers often need to read between the lines to get these messages. This implicit communication makes it hard for newcomers to the field and for those who with a peripheral interest in the literature to understand what is going on.
This article summarizes the various methodological problems we observe in the literature, aiming to be more explicit than prior work.

The ability to benchmark full autonomous driving systems in a manner that is reproducible across research laboratories first became possible around 2017 with the introduction of the [CARLA simulator](https://arxiv.org/abs/1711.03938). In the years ever since, various benchmarks have been proposed for CARLA and other simulators.
Before covering more specific mistakes made in CARLA benchmarks, we have to however address a bigger issue:

## NuScenes Planning
### Open-loop trajectory errors are unreliable.

[NuScenes](https://arxiv.org/abs/1903.11027) is a well-known autonomous driving dataset that features various computer vision benchmarks.
Using a dataset like nuScenes to evaluate the planning output of an autonomous driving stack is called open-loop planning, and works by feeding the sensor data in the stack and comparing its prediction with the ground truth action from the human driver in the dataset. This is called open-loop planning because the sensor input of future frames will not depend on the prediction of the driving stack, and neither will the other cars react to its behavior. Open-loop planning is appealing due to its computational efficiency compared to closed-loop evaluation (either in the real world or in simulation) and it does not introduce a sim-to-real gap with respect to the sensor data, unlike simulators.
Given these advantages, it was investigated early on whether the open-loop trajectory errors, measured via L2 loss, could act as a reliable performance indicator for closed-loop performance, which is what we ultimately want.
Unfortunately, this turned out not to be the case as was shown in [Codevilla et al. 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Felipe_Codevilla_On_Offline_Evaluation_ECCV_2018_paper.pdf). The open-loop L2 error did not necessarily correlate with the closed-loop error. In other words, the open-loop L2 loss can be misleading.
As a result, the community focused its benchmarking efforts on closed-loop simulation.
As researchers working on end-to-end autonomous driving solutions, we can confirm these results. In our experience, open-loop validation losses of end-to-end systems are not useful as indicators for closed-loop performance, and we therefore do not report them.

### The L2 metric became popular.
For the reason outlined before, the nuScenes paper itself did **not** propose a planning benchmark (according to private communication with the authors, intentionally so).
However, the planning community grew significantly around 2022/2023 and some researchers, seemingly unaware of these early results, began using the nuScenes dataset to benchmark planning using the L2 loss (or variants of this, called displacement errors) as a primary metric. This benchmark became known as nuScenes Planning. Due to the good reputation of nuScenes in the vision community (and one of the papers using the benchmark winning a prestigious vision award), this benchmark was widely adopted.

### NuScenes planning led to misleading results.
Several research papers have supplemented their nuScenes planning results with closed-loop CARLA simulation results. This made it evident that the nuScenes planning methods were behind the state-of-the-art methods from CARLA. Although, this conclusion is implicit as only older papers were reported as CARLA baselines.

In an attempt to see whether the previously discussed findings also apply to nuScenes planning in particular, [Zhai et al. 2023](https://arxiv.org/abs/2305.10430) introduced an innovative diagnostic test. They proposed a driving stack called AD-MLP that has no access to perception inputs and solely extrapolates past movement in the dataset, based on ego-status information. This approach is viable because of the open-loop nature of the dataset and the inherent smoothness of driving logs. To be precise here, AD-MLP is an explicitly unrealistic model whose purpose is to expose the flaws of a metric or benchmarks, analogous to a constant baseline as used in other machine learning fields.
It turned out that AD-MLP works exceptionally well in nuScenes planning. Even after the publication of numerous subsequent papers, it remains the state of the art on nuScenes planning. 
Unfortunately, these results were largely ignored in the nuScenes planning community initially, although [Li et al. 2024](https://arxiv.org/abs/2312.03031v2) later reproduced them in a more rigorous fashion.

Recent benchmarks like Bench2Drive or NAVSIM have adopted AD-MLP as a sanity check. AD-MLP having poor performance on a benchmark is an indicator for the quality of the benchmark.

### There are additional flaws in the nuScenes planning benchmark.

Besides these fundamental flaws in nuScenes planning, numerous execution-related issues regarding the benchmark have also been identified. [Weng et al. 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_PARA-Drive_Parallelized_Architecture_for_Real-time_Autonomous_Driving_CVPR_2024_paper.pdf) describes several of these issues, such as the metrics being incorrectly computed, inconsistencies in metric definitions across different papers and post-processing being done differently across methods. There are more problems than discussed in the paper. For example, the conditioning input command is computed based on the waypoint labels, resulting in label leakage. Furthermore, the size of the validation set is orders of magnitudes smaller than in other planning benchmarks (this limitation is specific to nuScenes planning, as perception tasks have many labels per frame).

### We need to stop using nuScenes planning!

In our view, the only rational course of action regarding nuScenes planning is to cease using it as a benchmark.
The numerous issues in the benchmark render it unsuitable for evaluating the performance of autonomous driving stacks.
If you are a reviewer evaluating a paper in 2025 that bases its claims solely on nuScenes planning results, we ask you to reject it.
If you are an author and encounter a reviewer (Reviewer 2, perhaps) requesting nuScenes planning results, we recommend politely declining. In such cases, you could refer to this text or the cited papers as justification for your decision.

### NuScenes planning reinvigorated the search for better open-loop metrics.

A positive outcome of the nuScenes planning story was that it highlighted the substantial interest in cost-effective open-loop evaluation and started a search for better open loop metrics ([Weng et al. 2024](https://xinshuoweng.github.io/paradrive/assets/camera_ready.pdf), [Li et al.2024](https://arxiv.org/abs/2312.03031), [Dauner et al. 2024](https://arxiv.org/abs/2406.15349)). In particular, the discovery that the PDM-score metric is correlated to closed-loop performance led to a new type of benchmark (NAVSIM). The PDM-score is an open-loop metric that performs a pseudo simulation based on the "world on rails" assumption.
NAVSIM represents a significant improvement over nuScenes planning, as an open-loop benchmark.
However, it remains unable to measure important issues such as compounding errors, implying that it should be complemented with closed-loop experiments. For research groups, lacking the computational resources required for large-scale closed-loop simulations, NAVSIM currently serves as the recommended alternative to nuScenes planning.

### Using nuScenes for perception tasks is valid.
It is important to note, that we do not claim that all conclusions drawn from nuScenes planning are incorrect. Rather, we claim that the validity of these ideas remain uncertain. While some of the proposed approaches may indeed be effective, others may not. To know, the community needs to reevaluate these ideas on appropriate and reliable benchmarks. Furthermore, our criticism is specific to nuScenes planning and does not extend to other nuScenes benchmarks for perception and prediction tasks.

## CARLA benchmarks
[CARLA](https://arxiv.org/abs/1711.03938) is an autonomous driving simulator developed using the Unreal Engine. It is capable of simulating a wide range of sensor data, ground truth labels, safety critical scenarios, and background traffic. Its open-source release in 2017 enabled the community to benchmark entire autonomous driving stacks in closed-loop. Due to the community's continual commitment to update and improve the simulator, it has become the de facto standard for rigorous evaluation of autonomous driving stacks.
CARLA, as a simulator, does not inherently include a benchmark nor a dataset for model development.
Instead, the community has developed numerous benchmarks based on the CARLA simulator, which increased in difficulty over time. Some noteworthy once include the [original CARLA benchmark](https://arxiv.org/abs/1711.03938), [NoCrash](https://arxiv.org/abs/1904.08980), [Town05 Short/Long](https://arxiv.org/abs/2104.09224), [LAV routes](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_From_All_Vehicles_CVPR_2022_paper.pdf), [Longest6](https://arxiv.org/abs/2205.15997), [CARLA leaderboard 1.0 test routes](https://leaderboard.carla.org/#leaderboard-10), [Bench2Drive](https://arxiv.org/abs/2406.03877), [CARLA leaderboard 2.0 validation routes](https://leaderboard.carla.org) and [CARLA leaderboard 2.0 test routes](https://leaderboard.carla.org). These benchmarks are approximately arranged in order of increasing difficulty, although the list is not exhaustive.

### Papers frequently have errors regarding the setup details of CARLA benchmarks.

These CARLA benchmarks have various setup conditions including the CARLA towns from which data collection is permitted, which safety critical scenarios to evaluate with, what routes to drive along, the traffic density to be used and more.
Inconsistencies between these parameters can undermine the validity of comparisons between methods.

A recurring issue in the literature is authors comparing numbers from different benchmarks to each other.
This is a methodological mistake akin to comparing accuracies on CiFAR-10 to accuracies on MNIST, rendering the comparison meaningless.

Furthermore, we observe subtle errors in the details of evaluations, such as training on validation towns, evaluating without safety critical scenarios, neglecting to adjust the traffic density, and citing the wrong papers. A notable example of misquotation involves the TransFuser model, where author often report the (weaker) performance from the earlier conference paper, but cite the (stronger) stack from the journal extension.
These mistakes are so widespread in the literature that caution is warranted when copying results from tables in other published works. Instead of directly using reported results, it is recommended to re-evaluate baselines on benchmarks, as many of the baselines are open-source. At a minimum, researchers should verify the correctness of the numbers they reference.

When conducting evaluations, ensure that the conditions for a proper benchmark setup are followed. Any deviations from these conditions should be documented and explained.

### Closed-loop (CARLA) benchmarks are sensitive to random chance.

Another notable challenge with closed-loop benchmarks is that results are fundamentally sensitive to random chance (the seed). It is a standard practice to repeat evaluations at least 3 times and average the result. This issue is particularly pronounced in end-to-end methods, which often exhibit high training variance. Consequently, it is sometimes necessary to evaluate multiple training seeds as well ([Prakash et al. 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prakash_Exploring_Data_Aggregation_in_Policy_Learning_for_Vision-Based_Urban_Autonomous_CVPR_2020_paper.pdf), [Behl et al. 2020](https://arxiv.org/abs/2005.10091), [Chitta et al. 2023](https://www.cvlibs.net/publications/Chitta2022PAMI.pdf)). For example, in experiments on the LAV benchmark, we found it necessary to average the result of 3 training seeds each evaluated 3 times, otherwise the results were heavily influenced by random chance.
Unfortunately, recent trends in conference publications have seen a resurgence of single-seed evaluations. As a reviewer, it is important to remind authors that evaluating at least 3 seeds is necessary to ensure reliable comparisons.

The large amount of different benchmarks also introduced two additional subtle issues:

### Some papers propose new benchmarks with misleading names.

Some authors have used the confusion in the field to introduce easier benchmarks under the same name as existing, more challenging benchmarks. This practice can artificially enhance the perceived performance of their methods. For instance, this has occurred multiple times with the CARLA leaderboard, due to its good reputation and high difficulty. Although such incidents are relatively infrequent, they do occur, necessitating vigilance from reviewers.

### Some papers incorrectly claim state of the art.

Another significant issue is the difficulty of accurately tracking the state of the art on the various benchmarks. This has led to instances where authors are making incorrect claims about their method being state of the art even though their results are clearly lagging behind by several years. Such claims are often obfuscated by exclusively reporting comparisons to outdated baselines. Several papers published at top-tier conferences in 2024 have made such incorrect claims.
Reviewers who are uncertain about the state of the art on a particular benchmark are encouraged to consult other experts, such as the author of the benchmark.
It is acceptable to claim state of the art in the presence of concurrent work, e.g. from arXiv within the last 6 months (roughly corresponding to one conference review cycle). A method that is outperformed by published work from three years earlier cannot be considered state of the art.

### Some papers omit performance critical details.

The CARLA leaderboard test routes (version 1.0 and 2.0) are benchmarks with secret test routes, where evaluation is conducted by submitting code to a third-party server operated by the CARLA team. Participating teams can then view and publicize their results through a dedicated website, [https://leaderboard.carla.org/leaderboard/](https://leaderboard.carla.org/leaderboard/). Although operating such an independent evaluation system is resource-intensive, it ensures fair benchmarking and ensures that the claimed results have been achieved. Over the past five years, this system has generally functioned effectively and facilitated the fast progress we observed on these benchmarks.

However, a notable issue arose with the CARLA leaderboard 1.0. While the evaluation system ensures that results have been achieved, it does not ensure is that authors accurately report how they achieved their results. As documented by [Jaeger et al. 2023, Appendix C.6](https://arxiv.org/abs/2306.07957), the top three methods on the CARLA leaderboard were found to be unreproducible. The models and code of these methods are open source, yet the published models produce significantly weaker results than those reported in the papers and displayed on the leaderboard website. Since the published code corresponds to the description provided by the papers, it is questionable whether the papers accurately describe the models that were used to achieve the state-of-the-art results. 

This problem is difficult to identify during the peer review process, as it only becomes apparent through extensive reproduction efforts. We encourage reviewers to demand the release of code alongside a paper. Such transparency allows issues to be identified and resolved post-publication. There is no scientific justification for withholding the code of a publication. Unfortunately, the current top method on the CARLA leaderboard 1.0 did not publish its code, and as a result, its claims have never been independently reproduced.

Progress has stopped on the CARLA leaderboard 1.0 for the last two years, likely due to this situation.
As the community transitions to the CARLA leaderboard 2.0 we hope this situation does not reoccur.
The recent [NAVSIM leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim) aims to mitigate such problems by requiring the release of source code and models, and enforces reproducibility by removing submissions that fail to meet these criteria.


## Proprietary benchmarks
A significant number of autonomous driving papers authored by industrial labs are not evaluated on publicly available datasets or benchmarks, nor do they release code, models or data.
Considering the challenges in assessing the validity of claims made even when public data is used, this practice can erode trust in these works. As readers, we can often only conclude that the presented idea might be of interest, but we cannot determine from the paper alone whether it outperforms existing approaches.

Reproducing ideas solely based on the description in the paper is often a complex and time-intensive task that can take months. Moreover, attempts at reproduction may fail, either because the described idea was not genuinely effective or because crucial technical details were omitted.
As a result, many autonomous driving papers originating from industry are never reproduced.
This can have the secondary effect that some promising ideas may not get the attention they deserve in terms of citations or community adoption.

### Reasons for withholding code:

As we have said before, there is no scientific justification for withholding code in the context of autonomous driving.
Nonetheless, there are reasons why authors may choose not to publicize their code.
1. **The presented idea is ineffective**: The authors may intentionally obscure this by making the paper hard to reproduce.
2. **Commercial interests**: The authors may seek to prevent competitors from utilizing their work to maintain a competitive advantage.
3. **Resource constraints**: Preparing code for publication requires additional effort, and the authors may lack the time or resources to do so.

The problem is that papers falling under the second and third categories will inadvertently get associated with papers from category one because there is no practical way to differentiate them.
To address this, we propose an alternative approach for the second and third groups that avoids withholding the code. 

### Publish code under a non-commercial license!

What we would like to point out is that **there is no need to publish your code with an open-source license**. From a scientific standpoint, it is sufficient to release code under a **non-commercial research license**. Such licenses permit researchers to use the software for research purposes (which may benefit the author's company) while preventing competitors from utilizing the software, unless they negotiate a different license.

This practice is already well-established for autonomous driving datasets. For example, the [nuScenes license](https://www.nuscenes.org/terms-of-use-commercial) follows this approach.
 Another example of such a license is the [Software Copyright License for non-commercial scientific research purposes](https://icon.is.tue.mpg.de/license.html) used by the Max Plank Institute for Intelligent Systems. 

Rather than keeping the code entirely private, we encourage authors to publish their code under such licenses.
For those concerned about the effort involved in cleaning their code, we recommend publishing it as is. Most licenses include disclaimers against warranties, and even unpolished code is more valuable to the research community than publishing none at all.


## The simulation argument
This text advocates for benchmarking autonomous driving systems using closed-loop simulations. A commonly raised concern is that simulations lack realism, implying that real data is essential to develop methods that work on actual vehicles. This critique often suggests that simulation-based benchmarking may produce misleading results. While there is some validity to this concern, it is indeed possible to construct an unrepresentative simulation, the claim is often treated as self-evident. However, we are not aware of empirical evidence that demonstrates that state-of-the-art simulators like CARLA or nuPlan have led to misleading results. The misleading outcomes discussed earlier stem from factors unrelated to the simulators themselves.

There are two counterarguments to the claim that simulators lack sufficient realism for reliable benchmarking:

### Driving simulators might already be realistic enough for benchmarking.
 The first is that contemporary simulators may already be realistic enough to compel researchers to develop robust general-purpose methods that also perform effectively on actual vehicles.
An illustrative example is the [CVPR 2024 NAVSIM challenge](https://arxiv.org/abs/2406.15349), conducted from March to May 2024. 
NAVSIM is a simulator that only uses real sensor data and as a result, is more realistic than CARLA in terms of its sensor data.

In the challenge, the organizers reproduced the TransFuser method on NAVSIM as a baseline. TransFuser is a method that was designed and developed entirely on synthetic CARLA data. Interestingly, the TransFuser baseline outperformed most of the 463 competing submissions. Furthermore, the top-performing solution was an extension of TransFuser. This suggests that methods developed on synthetic CARLA data can also excel on real-world data. Of course, this is anecdotal evidence, and NAVSIM is still a simulation.

### Sim-to-real transfer already works in other robotics fields.
The second argument is that in other robotics fields, in particular locomotion, agile flight, and indoor navigation, zero-shot sim-to-real transfer has already been achieved. This recent development, described as ["A quiet revolution in robotics"](https://www.youtube.com/watch?v=K09erFsOnxA), involve methods trained entirely in simulation that were able to significantly advance the state of the art on real-world robots. This represents an empirical refutation of the argument that simulations can not be realistic enough for developing effective methods. However, it is important to note that a comparable level of sim-to-real transfer has not yet been convincingly demonstrated in the domain of autonomous driving.
There is one company that publicly claimed to have built a driving simulation that exhibits similar capabilities. However, due to the lack of publicly available details it is impossible to evaluate the validity or extent of this claim.

### There is no justification to abandon simulators for benchmarking.
Currently, the claim that the lack of realism in contemporary simulations is a fundamental problem for benchmarking can not be entirely refuted. However, current evidence points suggests that simulations remain a valuable tool, so there is no justification to abandon simulations for benchmarking. Data-driven simulations like [NAVSIM](https://arxiv.org/abs/2406.15349), [nuPlan](https://arxiv.org/abs/2106.11810) (not to be confused with nuScenes planning) or [Waymax](https://arxiv.org/abs/2310.08710) offer a important complement to synthetic simulators like CARLA. However, they cannot replace synthetic simulators entirely, because current data-driven simulations are easier to solve due to their lack of long simulations and safety-critical scenarios. Performance metrics currently exceed 90% on data-driven simulators and remain below 10% on the most challenging CARLA benchmark.


## The need for public research: 
Approximately a decade ago, with the rise in popularity of neural networks, the field of autonomous driving was swept by enthusiasm, which gave birth to the myth that "The science is solved." This optimism was understandable at the time, when limitations of neural networks, such as shortcut learning, were not known yet, and the challenges posed by the long tail of driving scenarios were not yet well understood. However, 10 years and billions of dollars in investments later, it has become evident that this initial confidence was overly optimistic. 

Today, it could be argued that our current level of scientific understanding is sufficient to enable the deployment of Level 4 autonomous driving systems. Whether such systems can be built in a commercially viable manner remains uncertain. However, it is clear that achieving Level 5 autonomy will require fundamental scientific advancements. The economic and practical advantages of deploying a universal system (Level 5) compared to building region-specific solutions (Level 4) are likely to be enormous.

Public research offers a significant cost advantage over private research. Every dollar invested in public research contributes to the advancement of the entire field, benefiting all companies. In contrast, private research only benefits one company, leading to duplicated efforts as the same ideas are rediscovered multiple times.

It is essential to recognize that **the autonomous driving challenge is, above all, a competition against reality itself**. We need public research that adheres to rigorous scientific practices in order to overcome this challenge!

Bernhard Jaeger, Kashyap Chitta, Daniel Dauner, Katrin Renz, Andreas Geiger; 16.12.2024

PS: If you are aware of additional issues related to benchmarking autonomous driving that are not yet discussed here, we encourage you to get in touch (bernhard.jaeger@uni-tuebingen.de).
