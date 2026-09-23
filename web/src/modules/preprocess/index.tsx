import { useStore } from '../../stores/useStore'
import { AppShell } from '../../app/AppShell'
import { Empty, Panel } from '../../app/ui'
import { OntologyCanvas } from './OntologyCanvas'
import { MaterialisePanel } from './MaterialisePanel'
import {
  ColumnTypePanel,
  ComponentPanel,
  ConstraintPanel,
  NodeCurationPanel,
  SchemaPanel,
  SourcePanel,
} from './panels'

export function PreprocessModule() {
  const schema = useStore((s) => s.schema)

  return (
    <AppShell
      left={
        <>
          <SourcePanel />
          {/* Directly under the source, above the schema summary rather than
              below it: this is a decision, the summary is a report, and a real
              endpoint's summary is long enough (97 class chips on the one this
              was built against) to push a decision a full screen out of sight. */}
          <ComponentPanel />
          <SchemaPanel />
          <ConstraintPanel />
        </>
      }
      centre={
        schema ? (
          <>
            <OntologyCanvas />
            <MaterialisePanel />
          </>
        ) : (
          <Panel className="flex-1">
            <Empty icon="layers">
              Load a knowledge graph on the left — a bundled sample, an RDF file, or a SPARQL
              endpoint. The induced ontology appears here as a draggable diagram, and every property
              on it becomes a candidate causal variable you can keep or drop.
            </Empty>
          </Panel>
        )
      }
      right={
        <>
          <ColumnTypePanel />
          <NodeCurationPanel />
        </>
      }
    />
  )
}
