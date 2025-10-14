
multi_model_score = 1
score_obj = Score()
icd_score = score_obj.inter_cluster_dist()
cs_score = score_obj.cluster_stats()
per_model_score = icd_score * cs_score
num_of_clusters_coverage, assigned_clusters_coverage = score_obj.num_clusters_coverage(per_model_score)
multi_model_score = multi_model_score * per_model_score
num_of_clusters_elbow, assigned_clusters_elbow = score_obj.num_clusters_elbow(multi_model_score)



