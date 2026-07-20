# Plotting

```python
import pyrater.plotting as plot

fig, ax = plot.credibility_plot(model)
fig.savefig("credibility.png")

fig, ax = plot.posterior_plot(model, "difficulty")
```

Both functions return a plain `(Figure, Axes)` pair — nothing here ever
calls `plt.show()`, so you're free to keep customizing before saving or
embedding in a notebook.

`credibility_plot` is a jittered strip plot, one point per annotator, in
the style of the paper's Figure 1: the best and worst annotators are
highlighted, and optionally annotated via `label_fn`:

```python
plot.credibility_plot(model, label_fn=lambda j: data.annotator_ids[j])
```

`posterior_plot(model, quantity)` is a histogram of `"difficulty"` or
`"credibilities"`.
