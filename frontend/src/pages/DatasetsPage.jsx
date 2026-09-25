import { Link } from 'react-router-dom'
export default function DatasetsPage() { return <><header className="page-header"><p className="eyebrow">Data workspace</p><h1>Datasets</h1><p>Upload a new CSV from the experiment workspace, then inspect its verified analytics.</p></header><section className="history-panel"><Link to="/experiments/new">Upload dataset</Link></section></> }
