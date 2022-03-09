# GET THE NUMBER OF ML-RELATED FDA APPROVALS
args = commandArgs(trailingOnly=TRUE)
dir_base = args[1]

avail_packages = rownames(installed.packages())
lst_libs = c('stringr', 'rvest', 'zoo', 'dplyr', 'tidyr', 'ggplot2', 'cowplot')
for (lib in lst_libs) {
  if (!(lib %in% avail_packages)) {
    print(sprintf('Installing package: %s', lib))
    install.packages(lib, repos='http://cran.us.r-project.org')
  }
  library(lib, character.only = TRUE)
}


dir_figures = file.path(dir_base, 'figures')
if (!dir.exists(dir_figures)) {
  dir.create(dir_figures)
}
dir_data = file.path(dir_base, 'data')
if (!dir.exists(dir_data)) {
  dir.create(dir_data)
}


#############################
# ---- (1) SCRAPE DATA ---- #

sleep = 1
fn_path = file.path(dir_data, 'df_fda.csv')
col_elements = str_c('tr:nth-child(',seq(7, 25),') td')
# Base URL for FDA
url = 'https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices'
fda = read_html(url)
# From the table, extract all links
tab = fda %>% html_elements(':nth-child(1) td') %>% html_elements('a')
df_devices = tibble(idt=html_text2(tab), url=html_attr(tab,'href'))
n_devices = nrow(df_devices)

if (file.exists(fn_path)) {
  print('File already exists, loading')
  df = read.csv(fn_path)
} else {
  # Loop over each device and scrape
  holder = list()
  for (i in seq(n_devices)) {
    print(sprintf('Iteration %i of %i',i,n_devices))
    # break
    idt = unlist(df_devices[i,'idt'])
    url = unlist(df_devices[i, 'url'])
    page = read_html(url)
    # Extract LHS of column
    lhs = page %>% html_elements('tr th') %>% html_text2()
    if (any(str_detect(lhs, 'Supplements'))) {
      print('Skipping this row')
      next
    }
    n_lhs = length(lhs)
    lhs = lhs[which(lhs == 'Regulation Number'):n_lhs]
    # Extract RHS
    rhs = sapply(col_elements, function(x) html_text2(html_elements(page, x))) %>% 
      lapply(function(x) x[1]) %>% 
      unlist %>% 
      str_replace_all('\\r|\\||\\n|\\t','') %>% 
      str_trim()
    n_rhs = length(rhs)
    idx_rhs = min(which(str_detect(rhs,'^[0-9]')))
    rhs = rhs[idx_rhs:n_rhs]
    rhs = rhs[!is.na(rhs)]
    tmp_df = tibble(idt=idt, lhs=lhs, rhs=rhs)
    holder[[i]] = tmp_df
    Sys.sleep(sleep)
  }
  # Merge and save
  df = do.call('rbind', holder)
  write.csv(df, fn_path, row.names=FALSE)
}


#########################
# ---- (2) PROCESS ---- #

lhs_lvls = c(`Decision Date`='date_dec', `Date Received`='date_rec', 
             `Decision`='decision', `Regulation Medical Specialty`='specialty')

# (i) Clean up
df_res = df %>%
  filter(lhs %in% names(lhs_lvls)) %>% 
  mutate(lhs = recode(lhs, !!!lhs_lvls)) %>% 
  pivot_wider(id_cols='idt', names_from='lhs', values_from='rhs') %>% 
  mutate_at(vars(starts_with('date')), .funs=list(~as.Date(.,format='%m/%d/%Y'))) %>% 
  mutate(decision=str_split_fixed(decision,'\\s{2,}|\\s\\-',2)[,1]) %>% 
  mutate(decision=ifelse(decision=='substantially equivalent', 'se', decision)) %>% 
  mutate(year=as.integer(format(date_dec,'%Y')),month=as.integer(format(date_dec,'%m'))) %>%
  mutate(quarter=(month %/% 4) + 1)


# (ii) How many approvals based on substantial equivalence?
df_res %>% 
  filter(year >= 2013) %>% 
  pull(decision) %>% 
  table %>% sort %>% print


# (iii) Quarterly approvals
df_count = df_res %>% 
  filter(year >= 2012) %>% 
  mutate(date=as.Date(str_c(year,month,'01',sep='-'))) %>% 
  group_by(date) %>% count()
tmp = tibble(date=seq.Date(as.Date('2012-01-01'),max(df_count$date), by = '1 month'))
df_count = tmp %>%
  left_join(df_count) %>% 
  mutate(n=ifelse(is.na(n),0,n)) %>% 
  mutate(mu=rollmean(n, k=12, align='right', na.pad=T)) %>% 
  drop_na %>% 
  pivot_longer(c(n, mu)) %>% 
  mutate(name=ifelse(name == 'n', 'Actual', 'Rolling average (12-month)'))
  


#########################
# ---- (3) PROCESS ---- #

gg_fda = ggplot(df_count, aes(x=date, y=12*value, color=name)) + 
  geom_line() + theme_bw() + 
  labs(y='# FDA approvals (annualized)') +  
  scale_color_discrete(name='Adjustment') + 
  scale_x_date(date_breaks = '1 year', date_labels = '%Y-%m') + 
  theme(axis.title.x = element_blank(), axis.text.x=element_text(angle=90),
        legend.position = c(0.25,0.75))
save_plot(file.path(dir_figures, 'gg_fda.png'), gg_fda, base_width=5, base_height=3.5)



print('~~~ End of get_fda.R ~~~')
