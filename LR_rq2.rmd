---
title: "LR_rq1"
output: html_document
---

```{r}
library(tidyverse)
input = read_csv("RQ1_100_cleaned.csv")

```

```{r}
RQ1 <- input %>% 
  mutate(Drift = case_when(
    `Drift type` == "No Drift" ~ "No Drift",
    `Drift type` == "Sudden Shock" ~ "Shock",
    TRUE ~ "Drift"
  ))

```

```{r}
RQ1 %>% 
  filter(Drift != 'Shock') %>% 
  group_by(Drift, Algorithm, `Drift type`) %>% 
  summarise(mean = mean(SMAPE), std =  sd(SMAPE), min = min(SMAPE), max = max(SMAPE), n = n())
```

```{r}
LRno <- RQ1 %>% 
  filter(Drift == 'No Drift', Algorithm=='Linear regression')

LRD <-  RQ1 %>% 
  filter(Drift == 'Drift', Algorithm=='Linear regression')

t.test(LRno$SMAPE, LRD$SMAPE, var.equal = F, alternative = 'less')
  
  
```
```{r}
RQ1 %>% 
  group_by()
```

  
```{r}
no_shock <- RQ1 %>% 
  filter(Drift!='Shock', Algorithm=='Linear regression')


```

```{r}
inc_LR <- no_shock %>% 
  filter(`Drift type` =='Incremental Drift')

sudden_LR <- no_shock %>% 
  filter(`Drift type`=='Sudden Drift')
```

```{r}
t.test(sudden_LR$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(inc_LR$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
```


#############################################################################
Under here is sudden
###############################################################################

 
```{r}
sudden_LR_train <- sudden_LR %>% 
  filter(`Drift time`=="Training")

sudden_LR_traintest <- sudden_LR %>% 
  filter(`Drift time`=="Training/Test")

sudden_LR_test <- sudden_LR %>% 
  filter(`Drift time`=="Test")

t.test(sudden_LR_train$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(sudden_LR_traintest$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(sudden_LR_test$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')

```

```{r}
RQ1 %>% 
  filter(Drift != 'Shock') %>% 
  group_by(Drift, Algorithm, `Drift type`, `Drift time`) %>% 
  summarise(mean = mean(SMAPE), std =  sd(SMAPE), min = min(SMAPE), max = max(SMAPE), n = n())
```

```{r}
sudden_LR_small <- sudden_LR %>% 
  filter(`Drift magnitude`=='Small')

sudden_LR_large <- sudden_LR %>% 
  filter(`Drift magnitude`=='Large')

t.test(sudden_LR_small$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(sudden_LR_large$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
```
```{r}
RQ1 %>% 
  filter(Drift != 'Shock') %>% 
  group_by(Drift, Algorithm, `Drift type`, `Drift magnitude`) %>% 
  summarise(mean = mean(SMAPE), std =  sd(SMAPE), min = min(SMAPE), max = max(SMAPE), n = n())

```

```{r}
sudden_LR_important <- sudden_LR %>% 
  filter(`Variable importance`=="Important")

sudden_LR_medium <- sudden_LR %>% 
  filter(`Variable importance`=="Medium")

sudden_LR_unimportant <- sudden_LR %>% 
  filter(`Variable importance`=="Unimportant")

t.test(sudden_LR_important$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(sudden_LR_medium$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(sudden_LR_unimportant$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')

```
```{r}
RQ1 %>% 
  filter(Drift != 'Shock') %>% 
  group_by(Drift, Algorithm, `Drift type`, `Variable importance`) %>% 
  summarise(mean = mean(SMAPE), std =  sd(SMAPE), min = min(SMAPE), max = max(SMAPE), n = n())
```
 
 
```{r}
sudden_LR_driftingDropped <- sudden_LR %>% 
  filter(`Dropped drifting`=="TRUE")

sudden_LR_driftingNotDropped <- sudden_LR %>% 
  filter(`Dropped drifting`=="FALSE")

t.test(sudden_LR_driftingDropped$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(sudden_LR_driftingNotDropped$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')


```
```{r}
RQ1 %>% 
  filter(Drift != 'Shock') %>% 
  group_by(Drift, Algorithm, `Drift type`,  `Dropped drifting`) %>% 
  summarise(mean = mean(SMAPE), std =  sd(SMAPE), min = min(SMAPE), max = max(SMAPE), n = n())

```
 
#############################################################################
Under here is incremental
###############################################################################

 
```{r}
inc_LR_train <- inc_LR %>% 
  filter(`Drift time`=="Training")

inc_LR_traintest <- inc_LR %>% 
  filter(`Drift time`=="Training/Test")

inc_LR_test <- inc_LR %>% 
  filter(`Drift time`=="Test")

t.test(inc_LR_train$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(inc_LR_traintest$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(inc_LR_test$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')

```


```{r}
inc_LR_small <- inc_LR %>% 
  filter(`Drift magnitude`=='Small')

inc_LR_large <- inc_LR %>% 
  filter(`Drift magnitude`=='Large')

t.test(inc_LR_small$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(inc_LR_large$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
```

```{r}
inc_LR_important <- inc_LR %>% 
  filter(`Variable importance`=="Important")

inc_LR_medium <- inc_LR %>% 
  filter(`Variable importance`=="Medium")

inc_LR_unimportant <- inc_LR %>% 
  filter(`Variable importance`=="Unimportant")

t.test(inc_LR_important$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(inc_LR_medium$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(inc_LR_unimportant$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')

```
 
 
```{r}
inc_LR_driftingDropped <- inc_LR %>% 
  filter(`Dropped drifting`=="TRUE")

inc_LR_driftingNotDropped <- inc_LR %>% 
  filter(`Dropped drifting`=="FALSE")

t.test(inc_LR_driftingDropped$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')
t.test(inc_LR_driftingNotDropped$SMAPE, LRno$SMAPE, var.equal = F, alternative='greater')



```

