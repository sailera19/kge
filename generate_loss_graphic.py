from kge.job.trace import Trace

trace_path = "traces/text/trace_text_training_04765.yaml"

trace = Trace()
trace.load(trace_path)

df_epochs = trace.to_dataframe({"event": "epoch_completed"})
df_evals = trace.to_dataframe({"event": "eval_completed"})

print("halt")