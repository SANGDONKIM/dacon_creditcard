set.seed(42)
options(max.print = 150)

library(doFuture)
library(magrittr)
library(tidymodels)
library(parsnip)
library(dials)
library(tune)

all_cores <- parallel::detectCores(logical = FALSE)

registerDoFuture()
cl <- parallel::makeCluster(all_cores)
plan(cluster, workers = cl)

data("credit_data")

credit_data %<>%
    set_names(., tolower(names(.)))

glimpse(credit_data)

split <- initial_split(credit_data, prop = 0.80, strata = "status")

df_train <- training(split)
df_test  <- testing(split)

(train_cv <- vfold_cv(df_train, v = 5, repeats = 3, strata = "status"))

(engine <- rand_forest(
    mtry = 2,
    trees = 500, 
    min_n = 10
) %>% 
        set_mode("classification") %>% 
        set_engine("ranger"))



recipe <- df_train %>%
    recipe(status ~ .) %>%
    
    # Imputation: assigning NAs to a new level for categorical 
    # (that's good practice, but not needed here) and median imputation for numeric
    step_unknown(all_nominal(), -status) %>% 
    step_medianimpute(all_numeric()) %>%
    
    # Combining infrequent categorical levels and introducing a new level 
    # for prediction time (that's good practice, but not needed here)
    step_other(all_nominal(), -status, other = "infrequent_combined") %>%
    step_novel(all_nominal(), -status, new_level = "unrecorded_observation") %>%
    
    # Hot-encoding categorical variables
    step_dummy(all_nominal(), -status, one_hot = TRUE) %>%
    
    # Creating additional ratio variables - they typically make sense 
    # in credit scoring problems
    step_mutate(
        ratio_expenses_income = expenses / (income + 0.001),
        ratio_assets_income = assets / (income + 0.001),
        ratio_debt_income = debt / (income + 0.001),
        ratio_debt_assets = debt / (assets + 0.001),
        ratio_amout_price = amount / (price + 0.001)
    ) %>% 
    
    # Adding upsampling 
    step_upsample(status, over_ratio(range = c(0.8, 1.2), trans = NULL))


(grid <- grid_regular(
    over_ratio() %>% range_set(c(0.5, 1.5)),
    levels = 11
))

t_rec <- workflow() %>% 
    add_recipe(recipe) %>% 
    add_model(engine)

library(tictoc)
tic()
fits <- tune_grid(
    t_rec,
    model = engine,
    resamples = train_cv,
    #grid = grid,
    #perf = metric_set(roc_auc),
    #control = control_grid(save_pred = TRUE)
)
toc()

fits

devtools::install_github("konradsemsch/ggrapid")

?estimate(fits) %>% 
    arrange(desc(over_ratio))
