# GET NUMBER OF ARXIV PUBLICATIONS BY MONTHS
args = commandArgs(trailingOnly=TRUE)
dir_base = args[1]

avail_packages = rownames(installed.packages())
lst_libs = c('stringr', 'rvest', 'dplyr', 'tidyr', 'zoo', 'ggplot2', 'cowplot')
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
fn_path = file.path(dir_data, 'df_arxiv.csv')
year_seq = seq(12, 21, 1)
month_seq = seq(1, 12, 1)
category_seq = c('stat.ML', 'cs.AI', 'cs.LG')
n_perm = length(year_seq) * length(month_seq) * length(category_seq)

if (file.exists(fn_path)) {
  print('File already exists, loading')
  df = read.csv(fn_path)
  
} else {
  print('File does not exist, scraping data')
  holder = list()
  j = 0
  for (category in category_seq) {
    print(sprintf('--- Category: %s ---', category))
    for (year in year_seq) {
      for (month in month_seq) {
        j = j + 1
        print(sprintf('Year=%i, month=%i (%i of %i)',year,month,j,n_perm))
        # Set up the URL
        url = str_c('https://arxiv.org/list/', category, '/', year, str_pad(month, 2, pad='0'))
        page = read_html(url)
        txt = page %>% html_element('small') %>% html_text2()
        # Save for holder
        slice = data.frame(category=category, year=year, month=month, txt=txt)
        holder[[j]] = slice
        Sys.sleep(sleep)
      }
    }  
  }
  df = do.call('rbind', holder)
  write.csv(df, file=fn_path, row.names=FALSE)
}


#########################
# ---- (2) PROCESS ---- #

# (i) Clean up the text
df_date = df %>% 
  mutate(n=unlist(str_extract_all(txt, '[0-9]{1,4}\\sentries'))) %>% 
  mutate(n=as.integer(str_replace_all(n,'\\sentries',''))) %>% 
  select(-txt)

# (ii) Create a datetime variable
df_date = df_date %>% 
  mutate(date=as.Date(str_c(2000+year,month,'01',sep='-'))) %>% 
  select(-c(year, month))

# (iii) Calculate a total category
df_agg = df_date %>% group_by(date) %>%
  summarise(n=sum(n)) %>% 
  mutate(n12=n * 12) %>% 
  mutate(mu=rollmean(n12, k=12, align='right', na.pad=T)) %>% 
  drop_na() %>% 
  pivot_longer(c(n12, mu)) %>% 
  mutate(name=ifelse(name == 'n12', 'Actual', 'Rolling average (12-month)'))



######################
# ---- (3) PLOT ---- #

gg_arxiv = ggplot(df_agg, aes(x=date, y=value, color=name)) + 
  geom_line() + theme_bw() + 
  labs(y='# publications (annualized)') +  
  scale_color_discrete(name='Adjustment') + 
  scale_x_date(date_breaks = '1 year', date_labels = '%Y-%m') + 
  theme(axis.title.x = element_blank(), axis.text.x=element_text(angle=90),
        legend.position = c(0.25,0.75))
save_plot(file.path(dir_figures, 'gg_arxiv.png'), gg_arxiv, base_width=5, base_height=3.5)


print('~~~ End of get_arxiv.R ~~~')




