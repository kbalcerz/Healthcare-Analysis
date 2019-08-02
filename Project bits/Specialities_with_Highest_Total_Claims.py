stack1 = go.Bar(
    y=speciality_df.specialty_description,
    x=speciality_df.total_drug_cost/1000000.0,
    name='Total Drug Cost',
    orientation = 'h',
     marker=dict(
	color='rgb(112,211,188)'
    )
)

stack2 = go.Bar(
    y=speciality_df.specialty_description,
    x=speciality_df.total_day_supply/1000000.0,
    name='Total Day Supply',
    orientation = 'h',
     marker=dict(
	color='rgb(41,181, 127)'
    )
)


stack3 = go.Bar(
    y=speciality_df.specialty_description,
    x=speciality_df.total_claim_count/1000000.0,
    name='Total Claims',
    orientation = 'h',
     marker=dict(
	color='rgb(3,81,51)'
    )
)

data = [stack1, stack2, stack3]
layout = go.Layout(
    barmode='relative',
    title= 'Top 10 Specialities with Highest Total Claims',
    margin=dict(t = 50, b=50, l = 200),
    xaxis = dict(title = 'Total Count (millions)'),
    yaxis = dict(title = 'Speciality', tickangle = 45)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='speciality-highest-total-claims')
