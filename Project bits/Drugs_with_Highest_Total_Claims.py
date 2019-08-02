stack1 = go.Bar(
    x=drug_df.Drug,
    y=drug_df.total_claim_count / 1000000.0 ,
    name='Total Claims',
     marker=dict(
	color='rgb(0,36,71)'
    )
)
stack2 = go.Bar(
    x=drug_df.Drug,
    y=drug_df.total_day_supply / 1000000.0,
    name='Total Day Supply',
     marker=dict(
	color='rgb(49,130,189)'
    )
)
stack3 = go.Bar(
    x=drug_df.Drug,
    y=drug_df.total_drug_cost / 1000000.0,
    name='Total Drug Cost',
     marker=dict(
	color='rgb(204,204,204)'
    )
)

data = [stack1, stack2, stack3]
layout = go.Layout(
    barmode='group',
    title= 'Top 20 Drugs with Highest Total Claims',
    margin=dict(b=200),
    xaxis = dict(title = 'Drug',tickangle = 60),
    yaxis = dict(title = 'Total Count (millions)')
    
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='drugs-highest-total-claims')


