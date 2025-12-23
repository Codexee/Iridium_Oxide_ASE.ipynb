import qiskit
import qiskit_superstaq as qss
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("SUPERSTAQ_API_KEY")

provider = qss.SuperstaqProvider(api_key)
qc = qiskit.QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
qc.draw(output="mpl")
backend = provider.get_backend("ibmq_fez_qpu")

# Specify "dry-run" as the method to submit & run a Superstaq simulation
job = backend.run(qc, method="dry-run", shots=100)
result = job.result().get_counts()
print(result)