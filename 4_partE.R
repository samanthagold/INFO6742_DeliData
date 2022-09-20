# Graphing up the new measures 
library(tidyverse)


# Setup -------------------------------------------------------------------

in_fp =  "/Users/sammygold/Desktop/INFO6742/INFO6742_DeliData/part_e/"
data <- read_csv(paste0(in_fp, "convo_w_polite.csv")) %>% 
    rename_with(~str_replace(.x, "meta.", ""), .cols = everything()) 



# Graphs ------------------------------------------------------------------

# Quick scatter 
data %>%
    ggplot(aes(x = meta.prop_users_found, 
               y = meta.correct_ratio)) + 
    geom_point()

# Quick heatmap 

cor(data %>% select(-id)) %>% 
    as.data.frame() %>% 
    rownames_to_column() %>% 
    pivot_longer(cols = -rowname, 
                 names_to = "Measure", 
                 values_to = "Correlation") %>% 
    ggplot(aes(x = rowname, y = Measure, fill = Correlation)) + 
    geom_tile() + 
    geom_text(aes(label = round(Correlation, 2))) + 
    theme(axis.text.x = element_text(angle = 90)) + 
    xlab("Measure") + 
    ggtitle("Correlation Between Conversation Measures")

# Subset the data so that we are only looking at conversations that used at least 1 username 
# Within conversations that use at least 1 username, does more username mentioning = more fruitful discussions??? 
# Evidence is weak here, but we do see a negative correlation. 
cor(data %>% select(-id) %>% filter(prop_utt_w_usernames != 0)) %>% 
    as.data.frame() %>% 
    rownames_to_column() %>% 
    pivot_longer(cols = -rowname, 
                 names_to = "Measure", 
                 values_to = "Correlation") %>% 
    ggplot(aes(x = rowname, y = Measure, fill = Correlation)) + 
    geom_tile() + 
    geom_text(aes(label = round(Correlation, 2))) + 
    theme(axis.text.x = element_text(angle = 90)) + 
    xlab("Measure") + 
    viridis::scale_fill_viridis("magma") + 
    ggtitle("Correlations Between Conversation Measures - Restricted Sample")


# Ok great, adding in the entropy measure that Jinsook had added 










