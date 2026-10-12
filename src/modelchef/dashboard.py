"""Live, read-only TensorBoard view of saved ModelChef run metrics."""
import json
import math
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def scalars(values, prefix=''):
    for key, value in values.items():
        tag = f'{prefix}/{key}' if prefix else key
        if isinstance(value, dict):
            yield from scalars(value, tag)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            yield tag, value


def text_event(tag, text, step=0):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
    from tensorboard.compat.proto.tensor_pb2 import TensorProto
    from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
    metadata = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name='text'))
    tensor = TensorProto(dtype=7, string_val=[text.encode('utf-8')],
        tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
    return Event(wall_time=time.time(), step=step,
        summary=Summary(value=[Summary.Value(tag=tag, metadata=metadata, tensor=tensor)]))


class MetricsMirror:
    """Consume only complete JSONL records; a writer may be appending concurrently."""
    def __init__(self, folder, writer):
        self.folder, self.writer, self.offset = Path(folder), writer, 0
        self.final_written = False
        self.event_offset, self.optimizer_step, self.phase = 0, 0, 'Preparing run'
        self.latest_message = ''
        config = self.folder / 'config.json'
        if config.exists():
            self.writer.add_event(text_event('recipe/configuration',
                '```json\n' + config.read_text(encoding='utf-8') + '\n```'))
        self.writer.add_event(text_event('recipe/pipeline',
            'Configuration/hardware → acquisition → processing/splits → tokenizer → '
            'packing → model → training → held-out evaluation → generation/report.\n\n'
            'The activity panel follows saved run events; it does not infer unreported operations.'))

    def update(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        def write(values, step):
            summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)
                                     for tag, value in scalars(values) if tag != 'step'])
            self.writer.add_event(Event(wall_time=time.time(), step=step, summary=summary))

        count = 0
        activity_changed = False
        events = self.folder / 'events.jsonl'
        if events.exists():
            with events.open('rb') as source:
                source.seek(self.event_offset)
                while line := source.readline():
                    if not line.endswith(b'\n'):
                        break
                    event = json.loads(line)
                    message = event.get('message', '')
                    if event.get('stage'):
                        self.phase = str(event['stage'])
                    elif message.strip().startswith('==='):
                        self.phase = message.strip().strip('=').strip()
                    if not message.strip().startswith('==='):
                        self.latest_message = message
                    activity_changed = True
                    self.event_offset = source.tell()
        path = self.folder / 'metrics.jsonl'
        if path.exists():
            if path.stat().st_size < self.offset:
                raise ValueError('Metrics file was truncated; restart the dashboard to reload it')
            with path.open('rb') as source:
                source.seek(self.offset)
                while line := source.readline():
                    if not line.endswith(b'\n'):
                        break
                    record = json.loads(line)
                    write(record, record['step'])
                    self.optimizer_step = record['step']
                    self.offset = source.tell()
                    count += 1
        if activity_changed or count:
            label = (str(self.optimizer_step) if self.optimizer_step else
                     '0 — no optimizer updates recorded yet')
            self.writer.add_event(text_event('activity/current_status',
                f'**Pipeline stage:** {self.phase}\n\n**Optimizer step:** {label}\n\n'
                f'**Latest reported activity:** {self.latest_message}\n\n'
                'Pipeline stages describe preparation/training/evaluation phases. '
                'The TensorBoard step number counts optimizer updates, not log messages.',
                self.optimizer_step))
        report = self.folder / 'report.json'
        if report.exists() and not self.final_written:
            try:
                data = json.loads(report.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                data = None  # The training process may still be writing its report.
            if data and data.get('status') == 'completed':
                write({'final': {key: data[key] for key in ['validation', 'test', 'performance']
                                 if key in data}}, data.get('optimizer_steps', 0))
                if data.get('generation'):
                    self.writer.add_event(text_event('final/generated_sample', data['generation'],
                                                     data.get('optimizer_steps', 0)))
                self.final_written = True
        self.writer.flush()
        return count


def dashboard(folder, port=6006, once=False, open_browser=True):
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    folder = Path(folder).resolve()
    if not (folder / 'config.json').is_file():
        raise ValueError('Provide a ModelChef run directory containing config.json')
    if not 1 <= port <= 65535:
        raise ValueError('Dashboard port must be between 1 and 65535')
    if not once:
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError as error:
                raise ValueError(f'Port {port} is in use; choose another --port') from error
    logdir = folder / 'tensorboard/dashboard'
    writer = EventFileWriter(str(logdir))
    server = None
    try:
        mirror = MetricsMirror(folder, writer)
        count = mirror.update()
        print(f'Loaded {count} saved metric records. Monitoring {folder}', flush=True)
        if once:
            return
        with (logdir / 'server.log').open('a', encoding='utf-8') as server_log:
            server = subprocess.Popen([sys.executable, '-m', 'tensorboard.main',
                '--logdir', str(logdir), '--host', '127.0.0.1', '--port', str(port),
                '--reload_interval', '5'], stdout=server_log, stderr=server_log)
        url = f'http://127.0.0.1:{port}'
        print(f'Dashboard: {url}. Ctrl+C stops monitoring; training is unaffected.', flush=True)
        for _ in range(60):
            if server.poll() is not None:
                raise RuntimeError(f'TensorBoard exited; inspect {logdir / "server.log"}')
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    break
            time.sleep(1)
        else:
            raise RuntimeError(f'TensorBoard startup timed out; inspect {logdir / "server.log"}')
        if open_browser:
            webbrowser.open(url)
        while server.poll() is None:
            mirror.update()
            time.sleep(5)
        raise RuntimeError(f'TensorBoard exited; inspect {logdir / "server.log"}')
    finally:
        writer.close()
        if server and server.poll() is None:
            server.terminate()
            server.wait(timeout=10)
