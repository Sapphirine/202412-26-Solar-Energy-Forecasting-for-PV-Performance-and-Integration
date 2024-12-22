from model_scripts import model, analysis


# Defining main function
def main():
    date_string = "2022-12-13"
    cluster_id = 3
    model.build_cluster_model_and_forecast(cluster_id, date_string)
    analysis.cluster_points_analysis(cluster_id, date_string)

    cluster_id = 6
    model.build_cluster_model_and_forecast(cluster_id, date_string)
    analysis.cluster_points_analysis(cluster_id, date_string)
    


if __name__=="__main__":
    main()