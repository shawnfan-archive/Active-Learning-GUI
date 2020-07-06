<template>
    <v-dialog v-model="dialog" persistent max-width="1000">
        <v-card>
        <v-card-title class="headline">{{ formTitle }}</v-card-title>

        <!-- <v-card-text>Latest model: {{model_name}}</v-card-text> -->
        <v-card-actions>
            <v-btn outlined color="red darken-1" text v-on:click="toggleSubmissionDialog">Cancel</v-btn>

            <v-spacer></v-spacer>

            <v-btn
            outlined
            color="green darken-1"
            text
            v-on:click="onSubmit(true)"
            >Train model from scratch</v-btn>
            <v-btn outlined color="green darken-1" text v-on:click="onSubmit(false)">Retrain model</v-btn>
        </v-card-actions>
        </v-card>
    </v-dialog>
</template>

<script>
export default {
    components: {
    },
    props: {
        formTitle: {
            type: String,
        },
        dataToSubmit: { 
            type: Object
        }
    },
    data() {
        return {

        }
    }    
}
</script>